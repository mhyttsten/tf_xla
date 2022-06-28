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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSxla_hlo_profile_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSxla_hlo_profile_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSxla_hlo_profile_testDTcc() {
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

#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class HloProfileTest : public ClientLibraryTestBase {};

struct ParsedProfileOutputLine {
  int64_t cycles;
  std::string cycles_percentage;
  double usec;
  std::string flops;
  std::string trops;
  std::string bytes_per_sec;
  std::string bytes_per_cycle;
  std::string opcode;
};

::testing::AssertionResult HasFlops(
    const ParsedProfileOutputLine& parsed_line) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSxla_hlo_profile_testDTcc mht_0(mht_0_v, 225, "", "./tensorflow/compiler/xla/tests/xla_hlo_profile_test.cc", "HasFlops");

  if (RE2::FullMatch(parsed_line.flops, "[0-9.TGMk]+FLOP/s")) {
    return ::testing::AssertionSuccess()
           << "'flops' field present in  " << parsed_line.opcode << ": '"
           << parsed_line.flops << "'";
  }

  return ::testing::AssertionFailure()
         << "'flops' field absent in  " << parsed_line.opcode << ": '"
         << parsed_line.flops << "'";
}

::testing::AssertionResult HasTrops(
    const ParsedProfileOutputLine& parsed_line) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSxla_hlo_profile_testDTcc mht_1(mht_1_v, 241, "", "./tensorflow/compiler/xla/tests/xla_hlo_profile_test.cc", "HasTrops");

  if (RE2::FullMatch(parsed_line.trops, "[0-9.TGMk]+TROP/s")) {
    return ::testing::AssertionSuccess()
           << "'trops' field present in  " << parsed_line.opcode << ": '"
           << parsed_line.trops << "'";
  }

  return ::testing::AssertionFailure()
         << "'trops' field absent in  " << parsed_line.opcode << ": '"
         << parsed_line.trops << "'";
}

Status ParseOneProfileOutputLine(
    const std::string& line, bool expect_hlo,
    absl::flat_hash_map<std::string, ParsedProfileOutputLine>* parsed_results,
    absl::Span<const absl::string_view> opcodes_to_ignore = {}) {
  std::string separator = "[^:]*:: +";
  std::string match_percentage = R"(\d+\.\d*% +\d+Σ)";
  std::string match_cycles =
      R"((\d+) cycles +\( *()" + match_percentage + R"()\))";
  std::string match_usecs = "([0-9.]+) usec";
  std::string match_flops = "([^ ]*)";
  std::string match_trops = "([^ ]*)";
  std::string match_bytes_per_sec = "([0-9.TGMKi]*)(?:B/s)?";
  std::string match_bytes_per_cycle = "([0-9.TGMKi]*)(?:B/cycle)?";

  // The underlined part is what we're trying to match with match_opcode:
  //
  //   %dot33 = f32[256,256]{1,0} dot(...)
  //                              ^^^

  std::string match_opcode = expect_hlo ? "%[^=]+= [^ ]+ ([^(]+)\\(.*"
                                        : "(\\[total\\])( \\[entry\\])?";
  std::string regexp_pattern = absl::StrCat(
      " +", match_cycles, separator, match_usecs, separator, match_flops,
      separator, match_trops, separator, match_bytes_per_sec, separator,
      match_bytes_per_cycle, separator, match_opcode);

  ParsedProfileOutputLine parsed_line;
  bool matched = RE2::FullMatch(
      line, regexp_pattern, &parsed_line.cycles, &parsed_line.cycles_percentage,
      &parsed_line.usec, &parsed_line.flops, &parsed_line.trops,
      &parsed_line.bytes_per_sec, &parsed_line.bytes_per_cycle,
      &parsed_line.opcode);
  if (!matched) {
    return tensorflow::errors::InvalidArgument(
        "Input did not match regexp.  Input: ", line,
        ", Regexp: ", regexp_pattern);
  }

  if (!absl::c_linear_search(opcodes_to_ignore, parsed_line.opcode)) {
    InsertOrDie(parsed_results, parsed_line.opcode, parsed_line);
  }

  return Status::OK();
}

bool IsExtraMetricProfileOutputLine(const std::string& line) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("line: \"" + line + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSxla_hlo_profile_testDTcc mht_2(mht_2_v, 302, "", "./tensorflow/compiler/xla/tests/xla_hlo_profile_test.cc", "IsExtraMetricProfileOutputLine");

  return RE2::FullMatch(line, "Extra metric \\S+: \\d+");
}

// Returns void so that we can ASSERT.
void ExecuteAndFetchProfile(std::string* profile_output, LocalClient* client,
                            const XlaComputation& computation,
                            const Shape& lhs_arg_shape,
                            const Shape& rhs_arg_shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSxla_hlo_profile_testDTcc mht_3(mht_3_v, 313, "", "./tensorflow/compiler/xla/tests/xla_hlo_profile_test.cc", "ExecuteAndFetchProfile");

  LocalService* service = ClientLibrary::GetXlaService(client->platform());
  Backend* backend = service->mutable_backend();
  se::StreamExecutor* executor = backend->default_stream_executor();
  se::DeviceMemoryAllocator* allocator = backend->memory_allocator();
  auto* transfer_manager = backend->transfer_manager();
  TF_ASSERT_OK_AND_ASSIGN(
      StreamPool::Ptr stream_ptr,
      backend->BorrowStream(backend->default_device_ordinal()));

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer lhs_arg,
      transfer_manager->AllocateScopedShapedBuffer(
          lhs_arg_shape, allocator, backend->default_device_ordinal()));
  TF_ASSERT_OK(transfer_manager->TransferLiteralToDevice(
      stream_ptr.get(), Literal::CreateFromShape(lhs_arg_shape), lhs_arg));

  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer rhs_arg,
      transfer_manager->AllocateScopedShapedBuffer(
          rhs_arg_shape, allocator, backend->default_device_ordinal()));
  TF_ASSERT_OK(transfer_manager->TransferLiteralToDevice(
      stream_ptr.get(), Literal::CreateFromShape(rhs_arg_shape), rhs_arg));

  ExecutableBuildOptions build_options;
  build_options.mutable_debug_options()->set_xla_hlo_profile(true);
  TF_ASSERT_OK_AND_ASSIGN(
      auto local_executables,
      client->Compile(computation, {&lhs_arg_shape, &rhs_arg_shape},
                      build_options));

  Executable* executable = local_executables[0]->executable();
  HloExecutionProfile hlo_execution_profile(
      &executable->hlo_profile_printer_data(),
      &executable->hlo_profile_index_map());

  ExecutableRunOptions exec_run_options;
  exec_run_options.set_stream(stream_ptr.get());
  exec_run_options.set_allocator(backend->memory_allocator());
  exec_run_options.set_intra_op_thread_pool(
      backend->eigen_intra_op_thread_pool_device());
  ServiceExecutableRunOptions run_options(exec_run_options,
                                          /*borrow_stream=*/nullptr);
  std::vector<const ShapedBuffer*> args = {&lhs_arg, &rhs_arg};
  TF_ASSERT_OK_AND_ASSIGN(
      auto execution_result,
      executable->ExecuteOnStream(&run_options, args, &hlo_execution_profile));
  TF_ASSERT_OK(stream_ptr->BlockHostUntilDone());
  (void)execution_result;

  *profile_output = hlo_execution_profile.ToString(
      executor->GetDeviceDescription().clock_rate_ghz());

  XLA_VLOG_LINES(4, *profile_output);
}

XLA_TEST_F(HloProfileTest, DISABLED_ON_GPU(ProfileSingleComputation)) {
  const int64_t m = 256, k = 256, n = 256;
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {m, k});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {m, k});

  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetDefaultPlatform());
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client,
                          ClientLibrary::GetOrCreateLocalClient(platform));

  XlaBuilder builder(TestName());
  Tanh(Add(
      Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {m, k}), "dot_lhs"),
      Parameter(&builder, 1, ShapeUtil::MakeShape(F32, {k, n}), "dot_rhs")));

  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());

  std::string profile_output;
  ExecuteAndFetchProfile(&profile_output, client, computation, lhs_shape,
                         rhs_shape);
  VLOG(4) << "Profile Output:\n" << profile_output;
  std::vector<std::string> profile_output_lines =
      absl::StrSplit(profile_output, '\n');

  absl::flat_hash_map<std::string, ParsedProfileOutputLine>
      parsed_profile_lines;

  int line_no = 0;

  // Skip extra metrics.
  while (IsExtraMetricProfileOutputLine(profile_output_lines[line_no])) {
    line_no++;
  }

  line_no++;  // Skip 'Execution profile for ....'

  ASSERT_LT(line_no, profile_output_lines.size());
  TF_ASSERT_OK(ParseOneProfileOutputLine(profile_output_lines[line_no++],
                                         /*expect_hlo=*/false,
                                         &parsed_profile_lines));

  ASSERT_LT(line_no, profile_output_lines.size());
  TF_ASSERT_OK(ParseOneProfileOutputLine(profile_output_lines[line_no++],
                                         /*expect_hlo=*/true,
                                         &parsed_profile_lines));

  ASSERT_LT(line_no, profile_output_lines.size());
  TF_ASSERT_OK(ParseOneProfileOutputLine(profile_output_lines[line_no++],
                                         /*expect_hlo=*/true,
                                         &parsed_profile_lines));

  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine total_profile,
                          MaybeFind(parsed_profile_lines, "[total]"));
  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine dot_profile,
                          MaybeFind(parsed_profile_lines, "add"));
  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine tanh_profile,
                          MaybeFind(parsed_profile_lines, "tanh"));

  EXPECT_GT(total_profile.cycles, 0);
  EXPECT_EQ(total_profile.cycles_percentage, "100.% 100Σ");

  EXPECT_TRUE(HasFlops(total_profile));
  EXPECT_TRUE(HasTrops(total_profile));

  EXPECT_GT(total_profile.cycles, dot_profile.cycles);
  EXPECT_NE(dot_profile.cycles_percentage, "0.00%");
  EXPECT_NE(dot_profile.cycles_percentage, "100.00%");

  EXPECT_TRUE(HasFlops(dot_profile));
  EXPECT_FALSE(HasTrops(dot_profile));

  EXPECT_GT(total_profile.cycles, tanh_profile.cycles);
  EXPECT_NE(tanh_profile.cycles_percentage, "0.00%");
  EXPECT_NE(tanh_profile.cycles_percentage, "100.00%");

  EXPECT_FALSE(HasFlops(tanh_profile));
  EXPECT_TRUE(HasTrops(tanh_profile));
}

XLA_TEST_F(HloProfileTest, DISABLED_ON_GPU(ProfileWhileComputation)) {
  const int64_t size = 256;
  Shape matrix_shape = ShapeUtil::MakeShape(F32, {size, size});
  Shape while_result_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {}), matrix_shape});

  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          PlatformUtil::GetDefaultPlatform());
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client,
                          ClientLibrary::GetOrCreateLocalClient(platform));

  XlaComputation condition;
  {
    XlaBuilder builder("condition");
    auto state = Parameter(&builder, 0, while_result_shape, "state");
    auto iteration = GetTupleElement(state, 0);
    Gt(ConstantR0<int32_t>(&builder, 5), iteration);
    TF_ASSERT_OK_AND_ASSIGN(condition, builder.Build());
  }

  XlaComputation body;
  {
    XlaBuilder builder("body");
    auto state = Parameter(&builder, 0, while_result_shape, "state");
    auto matrix = GetTupleElement(state, 1);
    auto next_iteration =
        Add(GetTupleElement(state, 0), ConstantR0<int32_t>(&builder, 1));
    Tuple(&builder, {next_iteration, Mul(matrix, matrix)});
    TF_ASSERT_OK_AND_ASSIGN(body, builder.Build());
  }

  XlaBuilder builder(TestName());
  auto initial_while_state =
      Tuple(&builder, {ConstantR0<int32_t>(&builder, 0),
                       Parameter(&builder, 0, matrix_shape, "initial_value")});
  auto while_result = While(condition, body, initial_while_state);
  Add(GetTupleElement(while_result, 1),
      Parameter(&builder, 1, matrix_shape, "other_value"));

  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());

  std::string profile_output;
  ExecuteAndFetchProfile(&profile_output, client, computation, matrix_shape,
                         matrix_shape);
  SCOPED_TRACE(profile_output);

  std::vector<std::string> profile_output_lines =
      absl::StrSplit(profile_output, '\n');

  auto while_body_profile_start =
      absl::c_find_if(profile_output_lines, [](absl::string_view s) {
        return absl::StartsWith(s, "Execution profile for body");
      });

  ASSERT_NE(while_body_profile_start, profile_output_lines.cend());

  auto while_body_profile_end =
      std::find_if(while_body_profile_start, profile_output_lines.end(),
                   [](absl::string_view s) {
                     return absl::StartsWith(s, "********** microseconds ");
                   });

  // We emit a blank line before the "microseconds report" line.
  while_body_profile_end--;

  ASSERT_NE(while_body_profile_end, profile_output_lines.end());

  absl::flat_hash_map<std::string, ParsedProfileOutputLine>
      parsed_profile_lines;

  for (auto while_body_profile_i = while_body_profile_start + 1;
       while_body_profile_i != while_body_profile_end; while_body_profile_i++) {
    // There are multiple "get-tuple-element" instructions in the while body so
    // we ignore them -- we don't want parsed_profile_lines to be a multi-map.
    TF_ASSERT_OK(ParseOneProfileOutputLine(
        *while_body_profile_i,
        /*expect_hlo=*/while_body_profile_i != (while_body_profile_start + 1),
        &parsed_profile_lines, {"get-tuple-element"}));
  }

  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine total_while_body_profile,
                          MaybeFind(parsed_profile_lines, "[total]"));
  TF_ASSERT_OK_AND_ASSIGN(ParsedProfileOutputLine multiply_profile,
                          MaybeFind(parsed_profile_lines, "multiply"));

  EXPECT_GT(total_while_body_profile.cycles, 0);
  EXPECT_EQ(total_while_body_profile.opcode, "[total]");
  EXPECT_EQ(total_while_body_profile.cycles_percentage, "100.% 100Σ");

  EXPECT_GT(total_while_body_profile.cycles, multiply_profile.cycles);
  EXPECT_NE(multiply_profile.cycles_percentage, "0.00%");
  EXPECT_NE(multiply_profile.cycles_percentage, "100.00%");
}
}  // namespace
}  // namespace xla

static std::pair<int, char**> AddXlaHloProfileFlag(int argc, char** argv) {
  // Intentional "leak".
  char** new_argv = new char*[argc + 2];
  for (int i = 0; i < argc; i++) {
    new_argv[i] = argv[i];
  }

  // We do it this way (as opposed to piping in a modified DebugOptions
  // instance) for better end-to-end integration testing.
  new_argv[argc] = strdup("--xla_hlo_profile");

  // Fusion can change the Hlo instructions that show up in the final Hlo
  // executable, so block it here. Also block the WhileLoopInvariantCodeMotion
  // pass, otherwise a while loop is transformed and we could not match the
  // original name in the ProfileWhileComputation test.
  new_argv[argc + 1] = strdup(
      "--xla_disable_hlo_passes=fusion,fusion_merger,multi_output_fusion,"
      "while-loop-invariant-code-motion");
  return {argc + 2, new_argv};
}

GTEST_API_ int main(int argc, char** argv) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSxla_hlo_profile_testDTcc mht_4(mht_4_v, 568, "", "./tensorflow/compiler/xla/tests/xla_hlo_profile_test.cc", "main");

  std::vector<tensorflow::Flag> flag_list;
  xla::AppendDebugOptionsFlags(&flag_list);
  std::tie(argc, argv) = AddXlaHloProfileFlag(argc, argv);

  auto usage = tensorflow::Flags::Usage(argv[0], flag_list);
  if (!tensorflow::Flags::Parse(&argc, argv, flag_list)) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }

  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
