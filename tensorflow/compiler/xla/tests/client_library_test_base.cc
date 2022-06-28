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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc() {
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

#include "tensorflow/compiler/xla/tests/client_library_test_base.h"

#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

// Name of the interpreter backend.
constexpr char kInterpreter[] = "interpreter";

// Wrapper function that creates a nicer error message (than a bare
// ValueOrDie()) if the platform we intend to test is not available.
LocalClient* GetOrCreateLocalClientOrDie(
    const LocalClientOptions& client_options) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "GetOrCreateLocalClientOrDie");

  StatusOr<LocalClient*> result =
      ClientLibrary::GetOrCreateLocalClient(client_options);
  TF_CHECK_OK(result.status()) << " could not create local client for testing";
  return result.ValueOrDie();
}

// Helper functions to get the reference platform.
se::Platform* GetReferencePlatform() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_1(mht_1_v, 224, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "GetReferencePlatform");

  auto result = PlatformUtil::GetPlatform(kInterpreter);
  TF_CHECK_OK(result.status()) << "could not get interpreter platform";
  return result.ValueOrDie();
}

}  // namespace

ClientLibraryTestBase::ClientLibraryTestBase(
    se::Platform* platform, const LocalClientOptions& client_options)
    : client_(GetOrCreateLocalClientOrDie(client_options)),
      execution_options_(CreateDefaultExecutionOptions()) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_2(mht_2_v, 238, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ClientLibraryTestBase");

  CHECK_EQ(platform, client_options.platform());

  LocalClientOptions ref_options;
  ref_options.set_platform(GetReferencePlatform());
  ref_client_ = GetOrCreateLocalClientOrDie(ref_options);

  // Disabling constant_folding so that tests (usually written using Constants)
  // will exercise the intended code paths, instead of being constant folded.
  //
  // TODO(b/38354253): Constant folding is currently disabled. Change tests to
  // use Parameters instead of Constants, and re-enable constant folding by
  // default.
  execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
      "constant_folding");

  execution_options_.mutable_debug_options()
      ->set_xla_hlo_evaluator_use_fast_path(true);
}

ClientLibraryTestBase::ClientLibraryTestBase(se::Platform* platform)
    : execution_options_(CreateDefaultExecutionOptions()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_3(mht_3_v, 262, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ClientLibraryTestBase");

  LocalClientOptions default_options;
  default_options.set_platform(platform);
  client_ = GetOrCreateLocalClientOrDie(default_options);

  LocalClientOptions ref_options;
  ref_options.set_platform(GetReferencePlatform());
  ref_client_ = GetOrCreateLocalClientOrDie(ref_options);

  execution_options_.mutable_debug_options()->add_xla_disable_hlo_passes(
      "constant_folding");

  execution_options_.mutable_debug_options()
      ->set_xla_hlo_evaluator_use_fast_path(true);
}

std::string ClientLibraryTestBase::TestName() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_4(mht_4_v, 281, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::TestName");

  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

StatusOr<std::unique_ptr<GlobalData>> ClientLibraryTestBase::Execute(
    XlaBuilder* builder, absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_5(mht_5_v, 289, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::Execute");

  // Build the computation, as a convenience.
  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  return client_->Execute(computation, arguments, &execution_options_);
}

StatusOr<Literal> ClientLibraryTestBase::ExecuteAndTransfer(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const Shape* shape_with_output_layout) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_6(mht_6_v, 300, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ExecuteAndTransfer");

  ExecutionOptions execution_options = execution_options_;
  if (shape_with_output_layout != nullptr) {
    *execution_options.mutable_shape_with_output_layout() =
        shape_with_output_layout->ToProto();
  }
  return client_->ExecuteAndTransfer(computation, arguments,
                                     &execution_options);
}

StatusOr<Literal> ClientLibraryTestBase::ExecuteAndTransfer(
    XlaBuilder* builder, absl::Span<GlobalData* const> arguments,
    const Shape* shape_with_output_layout) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_7(mht_7_v, 315, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ExecuteAndTransfer");

  // Build the computation, as a convenience.
  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  return ExecuteAndTransfer(computation, arguments, shape_with_output_layout);
}

StatusOr<Literal> ClientLibraryTestBase::ExecuteAndTransferReference(
    const XlaComputation& computation, absl::Span<GlobalData* const> arguments,
    const Shape* shape_with_output_layout) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_8(mht_8_v, 326, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ExecuteAndTransferReference");

  ExecutionOptions execution_options = execution_options_;
  if (shape_with_output_layout != nullptr) {
    *execution_options.mutable_shape_with_output_layout() =
        shape_with_output_layout->ToProto();
  }
  execution_options.clear_device_handles();
  return ref_client_->ExecuteAndTransfer(computation, arguments,
                                         &execution_options);
}

std::string ClientLibraryTestBase::ExecuteToString(
    XlaBuilder* builder, absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_9(mht_9_v, 341, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ExecuteToString");

  auto computation_status = builder->Build();
  if (!computation_status.ok()) {
    return computation_status.status().ToString();
  }
  auto computation = computation_status.ConsumeValueOrDie();

  auto result =
      client_->ExecuteAndTransfer(computation, arguments, &execution_options_);
  if (!result.ok()) {
    return result.status().ToString();
  } else {
    return result.ValueOrDie().ToString();
  }
}

void ClientLibraryTestBase::ComputeAndCompareR1(
    XlaBuilder* builder, const tensorflow::core::Bitmap& expected,
    absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_10(mht_10_v, 362, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareR1");

  Literal expected_literal = LiteralUtil::CreateR1(expected);
  ClientLibraryTestBase::ComputeAndCompareLiteral(builder, expected_literal,
                                                  arguments);
}

void ClientLibraryTestBase::ComputeAndCompareLiteral(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments, const Shape* shape_with_layout) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_11(mht_11_v, 373, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareLiteral");

  EXPECT_IS_OK(ComputeAndCompareLiteralWithStatus(builder, expected, arguments,
                                                  shape_with_layout));
}

void ClientLibraryTestBase::ComputeAndCompareLiteral(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error,
    const Shape* shape_with_layout) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_12(mht_12_v, 384, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareLiteral");

  EXPECT_IS_OK(ComputeAndCompareLiteralWithStatus(builder, expected, arguments,
                                                  error, shape_with_layout));
}

Status ClientLibraryTestBase::ComputeAndCompareLiteralWithAllOutputLayouts(
    const xla::XlaComputation& computation, const Literal& expected,
    absl::Span<GlobalData* const> arguments,
    const std::function<void(const Literal& actual,
                             const std::string& error_message)>&
        verify_output) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_13(mht_13_v, 397, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareLiteralWithAllOutputLayouts");

  // Try with no layout requirement.
  TF_ASSIGN_OR_RETURN(auto actual, ExecuteAndTransfer(computation, arguments));
  verify_output(actual, "");

  // Try with all output layouts.
  std::vector<int64_t> minor_to_major(expected.shape().rank());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  do {
    auto layout = ShapeUtil::MakeShapeWithLayout(
        expected.shape().element_type(), expected.shape().dimensions(),
        minor_to_major);
    TF_ASSIGN_OR_RETURN(auto actual,
                        ExecuteAndTransfer(computation, arguments, &layout));
    verify_output(actual,
                  absl::StrCat("Test with output layout: ",
                               ShapeUtil::HumanStringWithLayout(layout)));
  } while (std::next_permutation(minor_to_major.begin(), minor_to_major.end()));
  return Status::OK();
}

Status ClientLibraryTestBase::ComputeAndCompareLiteralWithAllInputLayouts(
    const xla::XlaComputation& computation, const Literal& /*expected*/,
    absl::Span<GlobalData* const> arguments,
    const std::function<void(const Literal& actual,
                             const std::string& error_message)>& verify_output,
    const Shape* output_with_layout) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_14(mht_14_v, 426, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareLiteralWithAllInputLayouts");

  std::vector<GlobalData*> arguments_with_layout;
  std::vector<std::string> layout_strings;
  // This is a recursive function. It's an std::function instead of a lambda
  // because it needs to capture itself. The index is the index of the argument
  // to try all layouts for.
  std::function<Status(int64_t)> choose;
  choose = [&, this](int64_t index) -> Status {
    if (index < arguments.size()) {
      // Try out all layouts for the operand.
      TF_ASSIGN_OR_RETURN(auto literal,
                          client_->Transfer(*arguments[index], nullptr));
      // Skip tuples because they don't have a rank.
      if (literal.shape().IsTuple()) {
        layout_strings.push_back(
            ShapeUtil::HumanStringWithLayout(literal.shape()));
        arguments_with_layout.push_back(arguments[index]);
        TF_RETURN_IF_ERROR(choose(index + 1));
        arguments_with_layout.pop_back();
        layout_strings.pop_back();
        return Status::OK();
      }

      std::vector<int64_t> minor_to_major(literal.shape().rank());
      std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
      do {
        auto literal_relayout =
            literal.Relayout(LayoutUtil::MakeLayout(minor_to_major));
        layout_strings.push_back(
            ShapeUtil::HumanStringWithLayout(literal_relayout.shape()));
        TF_ASSIGN_OR_RETURN(auto data,
                            client_->TransferToServer(literal_relayout));
        arguments_with_layout.push_back(data.get());
        TF_RETURN_IF_ERROR(choose(index + 1));
        arguments_with_layout.pop_back();
        layout_strings.pop_back();
      } while (
          std::next_permutation(minor_to_major.begin(), minor_to_major.end()));
      return Status::OK();
    }

    // Every argument has an assigned layout.
    TF_ASSIGN_OR_RETURN(
        auto actual,
        ExecuteAndTransfer(computation,
                           absl::Span<GlobalData* const>(arguments_with_layout),
                           output_with_layout));
    std::string error_message = "Test with input layouts: ";
    for (const auto& str : layout_strings) {
      absl::StrAppend(&error_message, str, " ");
    }
    verify_output(actual, error_message);
    return Status::OK();
  };

  return choose(0);
}

StatusOr<Literal> ClientLibraryTestBase::ComputeAndTransfer(
    XlaBuilder* builder, absl::Span<GlobalData* const> arguments_passed_in,
    const Shape* shape_with_layout) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_15(mht_15_v, 489, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndTransfer");

  std::vector<GlobalData*> arguments(arguments_passed_in.begin(),
                                     arguments_passed_in.end());

  // Transfer and use elements of arguments_, if the AddParam() API was used.
  std::vector<std::unique_ptr<GlobalData>> owning_arguments;
  if (!arguments_.empty()) {
    CHECK(arguments.empty());
    for (const auto& argument : arguments_) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<GlobalData> owned_argument,
          client_->TransferToServer(MaybeConvertLiteralToBfloat16(argument)));
      owning_arguments.push_back(std::move(owned_argument));
      arguments.push_back(owning_arguments.back().get());
    }
  }

  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  return ExecuteAndTransfer(computation, arguments, shape_with_layout);
}

Status ClientLibraryTestBase::ComputeAndCompareLiteralWithStatus(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments_passed_in,
    const Shape* shape_with_layout) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_16(mht_16_v, 516, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareLiteralWithStatus");

  std::vector<GlobalData*> arguments(arguments_passed_in.begin(),
                                     arguments_passed_in.end());

  // Transfer and use elements of arguments_, if the AddParam() API was used.
  std::vector<std::unique_ptr<GlobalData>> owning_arguments;
  if (!arguments_.empty()) {
    CHECK(arguments.empty());
    for (const auto& argument : arguments_) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<GlobalData> owned_argument,
          client_->TransferToServer(MaybeConvertLiteralToBfloat16(argument)));
      owning_arguments.push_back(std::move(owned_argument));
      arguments.push_back(owning_arguments.back().get());
    }
  }

  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  if (ShapeUtil::ElementIsFloating(expected.shape()) ||
      ShapeUtil::ElementIsComplex(expected.shape())) {
    LOG(WARNING) << "performing exact comparison of floating point numbers";
  }
  // We allow using a float expected literal for a bfloat16 output. In this
  // case, we need to convert the expected literal to bfloat16.
  const Literal* expected_ptr = &expected;
  Literal converted_expected;
  Shape layout_shape;
  if (use_bfloat16_) {
    converted_expected = LiteralUtil::ConvertF32ToBF16(expected);
    expected_ptr = &converted_expected;
    if (shape_with_layout != nullptr) {
      layout_shape = *shape_with_layout;
      ShapeUtil::ForEachMutableSubshape(
          &layout_shape, [&](Shape* subshape, const ShapeIndex& /*index*/) {
            if (subshape->element_type() == F32) {
              subshape->set_element_type(BF16);
            }
          });
      shape_with_layout = &layout_shape;
    }
  }
  auto expect_equal = [&](const Literal& actual,
                          const std::string& error_message) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("error_message: \"" + error_message + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_17(mht_17_v, 562, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "lambda");

    EXPECT_TRUE(LiteralTestUtil::Equal(*expected_ptr, actual)) << error_message;
  };
  if (execution_options_.debug_options().xla_test_all_output_layouts()) {
    return ComputeAndCompareLiteralWithAllOutputLayouts(
        computation, *expected_ptr, arguments, expect_equal);
  }
  if (execution_options_.debug_options().xla_test_all_input_layouts()) {
    return ComputeAndCompareLiteralWithAllInputLayouts(
        computation, *expected_ptr, arguments, expect_equal, shape_with_layout);
  }
  TF_ASSIGN_OR_RETURN(auto actual, ExecuteAndTransfer(computation, arguments,
                                                      shape_with_layout));
  EXPECT_TRUE(LiteralTestUtil::Equal(*expected_ptr, actual));
  return Status::OK();
}

Status ClientLibraryTestBase::ComputeAndCompareLiteralWithStatus(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments_passed_in, ErrorSpec error,
    const Shape* shape_with_layout) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_18(mht_18_v, 585, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareLiteralWithStatus");

  std::vector<GlobalData*> arguments(arguments_passed_in.begin(),
                                     arguments_passed_in.end());

  // Transfer and use elements of arguments_, if the AddParam() API was used.
  std::vector<std::unique_ptr<GlobalData>> owning_arguments;
  if (!arguments_.empty()) {
    CHECK(arguments.empty());
    for (const auto& argument : arguments_) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<GlobalData> owned_argument,
          client_->TransferToServer(MaybeConvertLiteralToBfloat16(argument)));
      owning_arguments.push_back(std::move(owned_argument));
      arguments.push_back(owning_arguments.back().get());
    }
  }

  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());
  // We allow using a float expected literal for a bfloat16 output. In this
  // case, we need to convert the expected literal to bfloat16.
  const Literal* expected_ptr = &expected;
  Literal converted_expected;
  Shape layout_shape;
  if (use_bfloat16_) {
    converted_expected = LiteralUtil::ConvertF32ToBF16(expected);
    expected_ptr = &converted_expected;
    if (shape_with_layout != nullptr) {
      layout_shape = *shape_with_layout;
      ShapeUtil::ForEachMutableSubshape(
          &layout_shape, [&](Shape* subshape, const ShapeIndex& /*index*/) {
            if (subshape->element_type() == F32) {
              subshape->set_element_type(BF16);
            }
          });
      shape_with_layout = &layout_shape;
    }
  }
  auto expect_near = [&](const Literal& actual,
                         const std::string& error_message) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("error_message: \"" + error_message + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_19(mht_19_v, 627, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "lambda");

    EXPECT_TRUE(LiteralTestUtil::Near(*expected_ptr, actual, error))
        << error_message;
  };
  if (execution_options_.debug_options().xla_test_all_output_layouts()) {
    return ComputeAndCompareLiteralWithAllOutputLayouts(
        computation, *expected_ptr, arguments, expect_near);
  }
  if (execution_options_.debug_options().xla_test_all_input_layouts()) {
    return ComputeAndCompareLiteralWithAllInputLayouts(
        computation, *expected_ptr, arguments, expect_near, shape_with_layout);
  }
  TF_ASSIGN_OR_RETURN(auto actual, ExecuteAndTransfer(computation, arguments,
                                                      shape_with_layout));
  EXPECT_TRUE(LiteralTestUtil::Near(*expected_ptr, actual, error));
  return Status::OK();
}

void ClientLibraryTestBase::ComputeAndCompareR1U8(
    XlaBuilder* builder, absl::string_view expected,
    absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("expected: \"" + std::string(expected.data(), expected.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_20(mht_20_v, 651, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareR1U8");

  auto actual_status = ExecuteAndTransfer(builder, arguments);
  EXPECT_IS_OK(actual_status.status());
  if (!actual_status.ok()) {
    return;
  }
  auto actual = actual_status.ConsumeValueOrDie();

  // Turn the expected value into a literal.
  Literal expected_literal = LiteralUtil::CreateR1U8(expected);

  VLOG(1) << "expected: " << expected_literal.ToString();
  VLOG(1) << "actual:   " << actual.ToString();

  EXPECT_EQ(expected, actual.GetR1U8AsString());
}

void ClientLibraryTestBase::ComputeAndCompareTuple(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_21(mht_21_v, 673, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareTuple");

  auto actual_status = ExecuteAndTransfer(builder, arguments);
  EXPECT_IS_OK(actual_status.status());
  if (!actual_status.ok()) {
    return;
  }
  auto actual = actual_status.ConsumeValueOrDie();
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, actual));
}

void ClientLibraryTestBase::ComputeAndCompareTuple(
    XlaBuilder* builder, const Literal& expected,
    absl::Span<GlobalData* const> arguments, ErrorSpec error) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_22(mht_22_v, 688, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompareTuple");

  auto actual_status = ExecuteAndTransfer(builder, arguments);
  EXPECT_IS_OK(actual_status.status());
  if (!actual_status.ok()) {
    return;
  }
  auto actual = actual_status.ConsumeValueOrDie();
  EXPECT_TRUE(LiteralTestUtil::Near(expected, actual, error));
}

void ClientLibraryTestBase::ComputeAndCompare(
    XlaBuilder* builder, absl::Span<const Literal> arguments) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_23(mht_23_v, 702, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompare");

  auto status_or_data = ComputeValueAndReference(builder, arguments);
  EXPECT_IS_OK(status_or_data);
  if (!status_or_data.ok()) {
    return;
  }
  Literal reference, result;
  std::tie(reference, result) = status_or_data.ConsumeValueOrDie();
  EXPECT_TRUE(LiteralTestUtil::Equal(reference, result));
}

void ClientLibraryTestBase::ComputeAndCompare(
    XlaBuilder* builder, absl::Span<const Literal> arguments, ErrorSpec error) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_24(mht_24_v, 717, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::ComputeAndCompare");

  auto status_or_data = ComputeValueAndReference(builder, arguments);
  EXPECT_IS_OK(status_or_data);
  if (!status_or_data.ok()) {
    return;
  }
  Literal reference, result;
  std::tie(reference, result) = status_or_data.ConsumeValueOrDie();
  EXPECT_TRUE(LiteralTestUtil::Near(reference, result, error));
}

StatusOr<std::pair<Literal, Literal>>
ClientLibraryTestBase::ComputeValueAndReference(
    XlaBuilder* builder, absl::Span<const Literal> arguments) {
  // Transfer the arguments to the executor service. We put the unique_ptr's
  // into a vector to keep the data alive on the service until the end of this
  // function.
  std::vector<std::unique_ptr<GlobalData>> argument_data;
  std::vector<std::unique_ptr<GlobalData>> ref_argument_data;

  // Use `arguments_` if the AddParam() API was used.  Otherwise, use
  // plain `arguments`.
  if (!arguments_.empty()) {
    CHECK_EQ(arguments.size(), 0);
    arguments = arguments_;
  }

  for (const auto& arg : arguments) {
    TF_ASSIGN_OR_RETURN(auto data, client_->TransferToServer(arg.Clone()));
    TF_ASSIGN_OR_RETURN(auto ref_data, ref_client_->TransferToServer(arg));
    argument_data.push_back(std::move(data));
    ref_argument_data.push_back(std::move(ref_data));
  }

  // Create raw pointers to the GlobalData for the rest of the call stack.
  std::vector<GlobalData*> argument_data_ptr;
  std::transform(
      argument_data.begin(), argument_data.end(),
      std::back_inserter(argument_data_ptr),
      [](const std::unique_ptr<GlobalData>& data) { return data.get(); });
  std::vector<GlobalData*> ref_argument_data_ptr;
  std::transform(
      ref_argument_data.begin(), ref_argument_data.end(),
      std::back_inserter(ref_argument_data_ptr),
      [](const std::unique_ptr<GlobalData>& data) { return data.get(); });

  TF_ASSIGN_OR_RETURN(auto computation, builder->Build());

  TF_ASSIGN_OR_RETURN(auto result,
                      ExecuteAndTransfer(computation, argument_data_ptr));

  TF_ASSIGN_OR_RETURN(auto reference, ExecuteAndTransferReference(
                                          computation, ref_argument_data_ptr));

  return std::make_pair(std::move(reference), std::move(result));
}

XlaComputation ClientLibraryTestBase::CreateScalarRelu() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_25(mht_25_v, 777, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::CreateScalarRelu");

  XlaBuilder builder("relu");
  auto shape = ShapeUtil::MakeShape(use_bfloat16_ ? BF16 : F32, {});
  auto z_value = Parameter(&builder, 0, shape, "z_value");
  auto zero = use_bfloat16_
                  ? ConstantR0<bfloat16>(&builder, static_cast<bfloat16>(0.0f))
                  : ConstantR0<float>(&builder, 0.0f);
  Max(z_value, zero);
  auto computation_status = builder.Build();
  TF_CHECK_OK(computation_status.status());
  return computation_status.ConsumeValueOrDie();
}

XlaComputation ClientLibraryTestBase::CreateScalarMax() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_26(mht_26_v, 793, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::CreateScalarMax");

  XlaBuilder builder("max");
  auto shape = ShapeUtil::MakeShape(use_bfloat16_ ? BF16 : F32, {});
  auto x = Parameter(&builder, 0, shape, "x");
  auto y = Parameter(&builder, 1, shape, "y");
  Max(x, y);
  auto computation_status = builder.Build();
  TF_CHECK_OK(computation_status.status());
  return computation_status.ConsumeValueOrDie();
}

XlaComputation ClientLibraryTestBase::CreateScalarReluSensitivity() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_27(mht_27_v, 807, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::CreateScalarReluSensitivity");

  XlaBuilder builder("relu_sensitivity");
  auto shape = ShapeUtil::MakeShape(use_bfloat16_ ? BF16 : F32, {});
  auto activation = Parameter(&builder, 0, shape, "activation");
  auto backprop = Parameter(&builder, 1, shape, "backprop");
  auto zero = use_bfloat16_
                  ? ConstantR0<bfloat16>(&builder, static_cast<bfloat16>(0.0f))
                  : ConstantR0<float>(&builder, 0.0f);
  auto activation_gtz = Gt(activation, zero);
  Select(activation_gtz, /*on_true=*/backprop, /*on_false=*/zero);

  auto computation_status = builder.Build();
  TF_CHECK_OK(computation_status.status());
  return computation_status.ConsumeValueOrDie();
}

std::unique_ptr<Array2D<float>> ClientLibraryTestBase::CreatePatternedMatrix(
    int rows, int cols, float offset) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_28(mht_28_v, 827, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::CreatePatternedMatrix");

  auto array = absl::make_unique<Array2D<float>>(rows, cols);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      (*array)(row, col) = col + (row * 1000.0f) + offset;
    }
  }
  return array;
}

std::unique_ptr<Array2D<float>>
ClientLibraryTestBase::CreatePatternedMatrixWithZeroPadding(int rows, int cols,
                                                            int rows_padded,
                                                            int cols_padded) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_29(mht_29_v, 843, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::CreatePatternedMatrixWithZeroPadding");

  CHECK_GE(rows_padded, rows);
  CHECK_GE(cols_padded, cols);
  auto array = absl::make_unique<Array2D<float>>(rows_padded, cols_padded, 0.0);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < cols; ++col) {
      (*array)(row, col) = col + (row * 1000.0f);
    }
  }
  return array;
}

XlaOp ClientLibraryTestBase::AddParam(const Literal& argument,
                                      XlaBuilder* builder) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_30(mht_30_v, 859, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::AddParam");

  arguments_.push_back(argument.Clone());
  return Parameter(builder, /*parameter_number=*/arguments_.size() - 1,
                   MaybeConvertShapeToBfloat16(argument.shape()), "");
}

XlaOp ClientLibraryTestBase::CreateConstantFromLiteral(const Literal& literal,
                                                       XlaBuilder* builder) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_31(mht_31_v, 869, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::CreateConstantFromLiteral");

  return ConstantLiteral(builder, use_bfloat16_
                                      ? LiteralUtil::ConvertF32ToBF16(literal)
                                      : LiteralSlice(literal));
}

StatusOr<std::unique_ptr<GlobalData>>
ClientLibraryTestBase::CreateParameterAndTransferLiteral(
    int64_t parameter_number, const Literal& literal, const std::string& name,
    XlaBuilder* builder, XlaOp* data_handle) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_32(mht_32_v, 882, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::CreateParameterAndTransferLiteral");

  return CreateParameterAndTransferLiteral(parameter_number, literal, name,
                                           nullptr, builder, data_handle);
}

Shape ClientLibraryTestBase::MaybeConvertShapeToBfloat16(const Shape& shape) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_33(mht_33_v, 890, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::MaybeConvertShapeToBfloat16");

  if (!use_bfloat16_) {
    return shape;
  }
  Shape new_shape = shape;
  ShapeUtil::ForEachMutableSubshape(&new_shape,
                                    [](Shape* subshape, const ShapeIndex&) {
                                      if (subshape->element_type() == F32) {
                                        subshape->set_element_type(BF16);
                                      }
                                    });
  return new_shape;
}

Literal ClientLibraryTestBase::MaybeConvertLiteralToBfloat16(
    const Literal& literal) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_34(mht_34_v, 908, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::MaybeConvertLiteralToBfloat16");

  if (use_bfloat16_) {
    return LiteralUtil::ConvertF32ToBF16(literal);
  }
  return literal.Clone();
}

StatusOr<std::unique_ptr<GlobalData>>
ClientLibraryTestBase::CreateParameterAndTransferLiteral(
    int64_t parameter_number, const Literal& literal, const std::string& name,
    const DeviceHandle* device_handle, XlaBuilder* builder,
    XlaOp* data_handle) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSclient_library_test_baseDTcc mht_35(mht_35_v, 923, "", "./tensorflow/compiler/xla/tests/client_library_test_base.cc", "ClientLibraryTestBase::CreateParameterAndTransferLiteral");

  Literal param_literal = MaybeConvertLiteralToBfloat16(literal);
  TF_ASSIGN_OR_RETURN(auto data,
                      client_->TransferToServer(param_literal, device_handle));
  *data_handle =
      Parameter(builder, parameter_number, param_literal.shape(), name);
  return data;
}

}  // namespace xla
