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
class MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbuffer_donation_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbuffer_donation_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbuffer_donation_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

// This test runs a computation and reuses different subsets of
// input buffers as output buffers. The aliasing patterns executed
// are as follows:
// 1. output[0] == input[0], output[1] == input[1], output[2] == input[2]
// 2. output[0] == input[1], output[1] == input[2].
// 3. output[0] == input[2]
class BufferDonationTest : public HloTestBase {
 public:
  BufferDonationTest() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbuffer_donation_testDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/tests/buffer_donation_test.cc", "BufferDonationTest");

    client_ = ClientLibrary::LocalClientOrDie();
    backend_ = client_->mutable_backend();
    platform_ = backend_->platform();
    executor_ = backend_->default_stream_executor();
    TF_CHECK_OK(executor_->Init());
  }

 protected:
  LocalClient* client_;
  se::Platform* platform_;
  Backend* backend_;
  se::StreamExecutor* executor_;

  // If `donate_arguments` is `true` gives up ownership of the buffers used for
  // the input allocation.
  void RunAndCheck(std::unique_ptr<HloModule> hlo_module,
                   absl::Span<Literal const> argument_literals,
                   absl::Span<bool const> donate_arguments,
                   absl::Span<bool const> expected_runtime_aliasing,
                   const Literal& expected, std::string expected_failure = "") {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStestsPSbuffer_donation_testDTcc mht_1(mht_1_v, 236, "", "./tensorflow/compiler/xla/tests/buffer_donation_test.cc", "RunAndCheck");

    // Create a copy of the output shape because the HLO module is std::moved
    // into the compiler and may be deallocated.
    const Shape output_shape = hlo_module->result_shape();

    TF_ASSERT_OK_AND_ASSIGN(hlo_module, backend_->compiler()->RunHloPasses(
                                            std::move(hlo_module), executor_,
                                            /*device_allocator=*/nullptr));
    HloInputOutputAliasConfig alias_config =
        hlo_module->input_output_alias_config();
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        backend_->compiler()->RunBackend(std::move(hlo_module), executor_,
                                         /*device_allocator=*/nullptr));

    se::Stream stream(executor_);
    ASSERT_TRUE(stream.Init().ok());

    se::StreamExecutorMemoryAllocator memory_allocator(
        platform_, backend_->stream_executors());
    ExecutableRunOptions run_options;
    run_options.set_stream(&stream);
    run_options.set_allocator(&memory_allocator);
    ServiceExecutableRunOptions service_run_options(run_options,
                                                    backend_->StreamBorrower());

    std::vector<ExecutionInput> args;
    std::vector<ShapeTree<se::DeviceMemoryBase>> inputs_buffers;

    CHECK_EQ(argument_literals.size(), donate_arguments.size());

    for (int arg_num = 0; arg_num < argument_literals.size(); arg_num++) {
      const bool donate_argument = donate_arguments[arg_num];
      const Literal& argument_literal = argument_literals[arg_num];

      // Allocate input buffers that will be reused as outputs.
      TF_ASSERT_OK_AND_ASSIGN(
          ScopedShapedBuffer scoped_shaped_buffer,
          backend_->transfer_manager()->AllocateScopedShapedBuffer(
              argument_literal.shape(), &memory_allocator,
              executor_->device_ordinal()));
      ShapedBuffer shaped_buffer = scoped_shaped_buffer.release();
      TF_CHECK_OK(backend_->transfer_manager()->TransferLiteralToDevice(
          &stream, argument_literal, shaped_buffer));
      ShapeTree<se::DeviceMemoryBase> input_buffers = shaped_buffer.buffers();
      inputs_buffers.push_back(input_buffers);
      ShapeTree<MaybeOwningDeviceMemory> owned_buffers(
          argument_literal.shape());
      owned_buffers.ForEachMutableElement(
          [&](const ShapeIndex& index, MaybeOwningDeviceMemory* device_memory) {
            if (donate_argument) {
              *device_memory = se::OwningDeviceMemory(
                  input_buffers.element(index), executor_->device_ordinal(),
                  &memory_allocator);
            } else {
              *device_memory = input_buffers.element(index);
            }
          });

      args.emplace_back(ExecutionInput(std::move(owned_buffers)));
    }

    StatusOr<ExecutionOutput> output_status =
        executable->ExecuteAsyncOnStream(&service_run_options, std::move(args),
                                         /*hlo_execution_profile=*/nullptr);
    if (!expected_failure.empty()) {
      ASSERT_FALSE(output_status.ok());
      ASSERT_TRUE(absl::StrContains(output_status.status().error_message(),
                                    expected_failure))
          << "got: \n"
          << output_status.status().error_message() << " \nvs want\n"
          << expected_failure;
      return;
    }
    ExecutionOutput output = output_status.ConsumeValueOrDie();

    se::DeviceMemoryBase result_root_buffer = output.Result().root_buffer();
    LOG(INFO) << "result allocation = " << result_root_buffer.opaque()
              << "             size = " << result_root_buffer.size();

    // Check for expected aliasing between input and output buffers.
#ifndef XLA_TEST_BACKEND_INTERPRETER
    alias_config.ForEachAlias(
        [&](const ShapeIndex& output_index,
            const HloInputOutputAliasConfig::Alias& alias) {
          int arg_num = alias.parameter_number;
          const void* input_ptr =
              inputs_buffers[arg_num].element(alias.parameter_index).opaque();
          const void* output_ptr =
              output.Result().buffer(output_index).opaque();
          ASSERT_EQ(input_ptr == output_ptr,
                    expected_runtime_aliasing[arg_num]);
        });
#endif

    TF_ASSERT_OK(run_options.stream()->BlockHostUntilDone());
    TF_ASSERT_OK_AND_ASSIGN(
        Literal result_literal,
        backend_->transfer_manager()->TransferLiteralFromDevice(
            &stream, output.Result()));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result_literal));

    // Memories are automatically deallocated.
  }

  // Builds a simple compare-to-limit (x < 4) computation for a While.
  //
  // condition:
  //   const4[s32] -----------------------------------\
  //                                                   \
  //   param[(s32,f32[4])] --- get-tuple-element[0] --- less-than
  //
  std::unique_ptr<HloComputation> BuildWhileConditionComputation(
      const std::string& name) {
    auto builder = HloComputation::Builder(name);
    auto const4 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(4)));
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, t_s32_f32v1_, "x"));
    auto index = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(const4->shape(), param, 0));
    builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), index,
                                      const4, ComparisonDirection::kLt));
    return builder.Build();
  }

  // Builds a simple body computation for a While.
  //
  // body:
  //   constv[f32[1]] --------------------------------------\
  //                                                         \
  //                           /--- get-tuple-elementv[1] --- addv ---\
  //   param[(s32,f32[1])] ---|                                    tuple
  //                           \--- get-tuple-elementc[0] --- addc ---/
  //                                                         /
  //   const1[s32] -----------------------------------------/
  //
  std::unique_ptr<HloComputation> BuildWhileBodyComputation(
      const std::string& name) {
    auto builder = HloComputation::Builder(name);
    auto const1 = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(1)));
    auto constv = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({1.1f})));
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, t_s32_f32v1_, "x"));
    auto indexc = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(const1->shape(), param, 0));
    auto addc = builder.AddInstruction(HloInstruction::CreateBinary(
        indexc->shape(), HloOpcode::kAdd, indexc, const1));
    auto indexv = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(constv->shape(), param, 1));
    auto addv = builder.AddInstruction(HloInstruction::CreateBinary(
        constv->shape(), HloOpcode::kAdd, indexv, constv));
    builder.AddInstruction(HloInstruction::CreateTuple({addc, addv}));
    return builder.Build();
  }

  std::unique_ptr<HloModule> CreateTestModule(absl::string_view module_name) {
    std::unique_ptr<HloModule> module =
        CreateNewVerifiedModule(std::string(module_name));
    HloComputation* condition =
        module->AddEmbeddedComputation(BuildWhileConditionComputation("if<4"));
    HloComputation* body =
        module->AddEmbeddedComputation(BuildWhileBodyComputation("add-update"));

    HloComputation::Builder builder = HloComputation::Builder("SimpleWhile");
    HloInstruction* param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, t_s32_f32v1_, "param"));
    HloInstruction* while0 = builder.AddInstruction(
        HloInstruction::CreateWhile(t_s32_f32v1_, condition, body, param));
    HloInstruction* gte0 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(s32_, while0, 0));
    HloInstruction* gte1 = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(f32v1_, while0, 1));
    builder.AddInstruction(HloInstruction::CreateTuple({gte0, gte1}));
    module->AddEntryComputation(builder.Build());
    return module;
  }

  Shape s32_ = ShapeUtil::MakeShape(xla::S32, {});
  Shape r0f32_ = ShapeUtil::MakeShape(xla::F32, {});
  Shape f32v1_ = ShapeUtil::MakeShape(F32, {1});
  Shape t_s32_f32v1_ = ShapeUtil::MakeTupleShape({s32_, f32v1_});
};

// This tests a simple while loop where the parameters are aliased with the
// output buffers.
TEST_F(BufferDonationTest, SimpleWhileTupleTest) {
  std::unique_ptr<HloModule> module = CreateTestModule("SimpleWhile");
  TF_ASSERT_OK(module->input_output_alias_config().SetUpAlias({0}, 0, {0}));
  TF_ASSERT_OK(module->input_output_alias_config().SetUpAlias({1}, 0, {1}));

  std::vector<Literal> args;
  args.push_back(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<int>(0), LiteralUtil::CreateR1<float>({1.1f})}));
  Literal expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<int>(4), LiteralUtil::CreateR1<float>({5.5f})});
  RunAndCheck(std::move(module), args, /*donate_arguments=*/{true},
              /*expected_runtime_aliasing=*/{true}, expected);
}

// Tests a case where we have promised aliasing to the compiler, but the runtime
// has not actually donated the buffers.
TEST_F(BufferDonationTest, SimpleWhileTupleTestCopyProtection) {
  std::unique_ptr<HloModule> module =
      CreateTestModule("SimpleWhileCopyProtection");
  TF_ASSERT_OK(module->input_output_alias_config().SetUpAlias({0}, 0, {0}));
  TF_ASSERT_OK(module->input_output_alias_config().SetUpAlias({1}, 0, {1}));

  std::vector<Literal> args;
  args.push_back(LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<int>(0), LiteralUtil::CreateR1<float>({1.1f})}));
  Literal expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<int>(4), LiteralUtil::CreateR1<float>({5.5f})});
  RunAndCheck(std::move(module), args, /*donate_arguments=*/{false},
              /*expected_runtime_aliasing=*/{false}, expected);
}

// Tests a case that on XLA:GPU alias passthrough params automatically aliases
// pass-through parameters, even if the underlying buffer is not donated.
TEST_F(BufferDonationTest, TestNoCopyProtectionOnPassthroughParam) {
  HloModuleConfig config;
  config.set_alias_passthrough_params(true);

  StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY entry {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = (f32[], f32[]) tuple(a, b)
}
  )",
                                   config);

  std::vector<Literal> args;
  args.push_back(LiteralUtil::CreateR0<float>(0.1));
  args.push_back(LiteralUtil::CreateR0<float>(0.2));
  Literal expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(0.1), LiteralUtil::CreateR0<float>(0.2)});

  // Alias-passthrough-params is only implemented on GPU.
#ifdef XLA_TEST_BACKEND_GPU
  RunAndCheck(std::move(*module), args, /*donate_arguments=*/{false, false},
              /*expected_runtime_aliasing=*/{true, true}, expected);
#endif
}

TEST_F(BufferDonationTest, TestMustAliasNotDonated) {
  HloModuleConfig config;

  StatusOr<std::unique_ptr<VerifiedHloModule>> module =
      ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY entry {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT out = (f32[], f32[]) tuple(a, b)
}
  )",
                                   config);

  TF_ASSERT_OK(module->get()->input_output_alias_config().SetUpAlias(
      {0}, 0, {}, HloInputOutputAliasConfig::kMustAlias));

  std::vector<Literal> args;
  args.push_back(LiteralUtil::CreateR0<float>(0.1));
  args.push_back(LiteralUtil::CreateR0<float>(0.2));
  Literal expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(0.1), LiteralUtil::CreateR0<float>(0.2)});

#ifndef XLA_TEST_BACKEND_INTERPRETER
  RunAndCheck(std::move(*module), args,
              /*donate_arguments=*/{false, false}, {true, false}, expected,
              "An input was configured to be must-alias at "
              "compile time but not donated at runtime:");
#endif
}

}  // namespace
}  // namespace xla
