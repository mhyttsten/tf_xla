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
class MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc() {
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
#include "tensorflow/lite/kernels/test_util.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/nnapi/acceleration_test_util.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/acceleration_test_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_delegate_providers.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/simple_planner.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/tools/versioning/op_version.h"
#include "tensorflow/lite/version.h"

namespace tflite {

using ::testing::FloatNear;
using ::testing::Matcher;

std::vector<Matcher<float>> ArrayFloatNear(const std::vector<float>& values,
                                           float max_abs_error) {
  std::vector<Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(FloatNear(v, max_abs_error));
  }
  return matchers;
}

std::vector<Matcher<std::complex<float>>> ArrayComplex64Near(
    const std::vector<std::complex<float>>& values, float max_abs_error) {
  std::vector<Matcher<std::complex<float>>> matchers;
  matchers.reserve(values.size());
  for (const std::complex<float>& v : values) {
    matchers.emplace_back(
        AllOf(::testing::Property(&std::complex<float>::real,
                                  FloatNear(v.real(), max_abs_error)),
              ::testing::Property(&std::complex<float>::imag,
                                  FloatNear(v.imag(), max_abs_error))));
  }
  return matchers;
}

int SingleOpModel::AddInput(const TensorData& t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_0(mht_0_v, 253, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::AddInput");

  int id = 0;
  if (t.per_channel_quantization) {
    id = AddTensorPerChannelQuant(t);
  } else {
    id = AddTensor<float>(t, {});
  }
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddVariableInput(const TensorData& t) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_1(mht_1_v, 267, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::AddVariableInput");

  int id = 0;
  if (t.per_channel_quantization) {
    id = AddTensorPerChannelQuant(t);
  } else {
    id = AddTensor<float>(t, {}, true);
  }
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddIntermediate(TensorType type,
                                   const std::vector<float>& scale,
                                   const std::vector<int64_t>& zero_point) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_2(mht_2_v, 283, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::AddIntermediate");

  // Currently supports only int16 intermediate types.
  int id = tensors_.size();
  flatbuffers::Offset<QuantizationParameters> q_params =
      CreateQuantizationParameters(builder_, /*min=*/0, /*max=*/0,
                                   builder_.CreateVector<float>(scale),
                                   builder_.CreateVector<int64_t>(zero_point));
  tensors_.push_back(CreateTensor(builder_, builder_.CreateVector<int>({}),
                                  type,
                                  /*buffer=*/0,
                                  /*name=*/0, q_params, false));
  intermediates_.push_back(id);
  return id;
}

int SingleOpModel::AddNullInput() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_3(mht_3_v, 301, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::AddNullInput");

  int id = kTfLiteOptionalTensor;
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddOutput(const TensorData& t) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_4(mht_4_v, 310, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::AddOutput");

  int id = 0;
  if (t.per_channel_quantization) {
    id = AddTensorPerChannelQuant(t);
  } else {
    id = AddTensor<float>(t, {});
  }
  outputs_.push_back(id);
  return id;
}

void SingleOpModel::SetBuiltinOp(BuiltinOperator type,
                                 BuiltinOptions builtin_options_type,
                                 flatbuffers::Offset<void> builtin_options) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_5(mht_5_v, 326, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::SetBuiltinOp");

  opcodes_.push_back(CreateOperatorCode(builder_, type, 0, 0));
  operators_.push_back(CreateOperator(
      builder_, /*opcode_index=*/0, builder_.CreateVector<int32_t>(inputs_),
      builder_.CreateVector<int32_t>(outputs_), builtin_options_type,
      builtin_options,
      /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS, 0,
      builder_.CreateVector<int32_t>(intermediates_)));
}

void SingleOpModel::SetCustomOp(
    const string& name, const std::vector<uint8_t>& custom_option,
    const std::function<TfLiteRegistration*()>& registration) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_6(mht_6_v, 342, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::SetCustomOp");

  custom_registrations_[name] = registration;
  opcodes_.push_back(
      CreateOperatorCodeDirect(builder_, BuiltinOperator_CUSTOM, name.data()));
  operators_.push_back(CreateOperator(
      builder_, /*opcode_index=*/0, builder_.CreateVector<int32_t>(inputs_),
      builder_.CreateVector<int32_t>(outputs_), BuiltinOptions_NONE, 0,
      builder_.CreateVector<uint8_t>(custom_option),
      CustomOptionsFormat_FLEXBUFFERS));
}

void SingleOpModel::AllocateAndDelegate(bool apply_delegate) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_7(mht_7_v, 356, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::AllocateAndDelegate");

  CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
      << "Cannot allocate tensors";
  interpreter_->ResetVariableTensors();

  // In some rare cases a test may need to postpone modifying the graph with
  // a delegate, e.g. if tensors are not fully specified. In such cases the
  // test has to explicitly call ApplyDelegate() when necessary.
  if (apply_delegate) ApplyDelegate();
}

void SingleOpModel::BuildInterpreter(std::vector<std::vector<int>> input_shapes,
                                     int num_threads,
                                     bool allow_fp32_relax_to_fp16,
                                     bool apply_delegate,
                                     bool allocate_and_delegate) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_8(mht_8_v, 374, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::BuildInterpreter");

  input_shapes_ = input_shapes;
  allow_fp32_relax_to_fp16_ = allow_fp32_relax_to_fp16;
  apply_delegate_ = apply_delegate;
  allocate_and_delegate_ = allocate_and_delegate;

  auto opcodes = builder_.CreateVector(opcodes_);
  auto operators = builder_.CreateVector(operators_);
  auto tensors = builder_.CreateVector(tensors_);
  auto inputs = builder_.CreateVector<int32_t>(inputs_);
  auto outputs = builder_.CreateVector<int32_t>(outputs_);
  // Create a single subgraph
  std::vector<flatbuffers::Offset<SubGraph>> subgraphs;
  auto subgraph = CreateSubGraph(builder_, tensors, inputs, outputs, operators);
  subgraphs.push_back(subgraph);
  auto subgraphs_flatbuffer = builder_.CreateVector(subgraphs);

  auto buffers = builder_.CreateVector(buffers_);
  auto description = builder_.CreateString("programmatic model");
  builder_.Finish(CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                              subgraphs_flatbuffer, description, buffers));

  uint8_t* buffer_pointer = builder_.GetBufferPointer();
  UpdateOpVersion(buffer_pointer);

  bool use_simple_allocator =
      tflite::KernelTestDelegateProviders::Get()->ConstParams().Get<bool>(
          tflite::KernelTestDelegateProviders::kUseSimpleAllocator);

  if (!resolver_) {
    if (!bypass_default_delegates_) {
      // Check if any delegates are specified via the commandline flags. We also
      // assume the intention of the test is to test against a particular
      // delegate, hence bypassing applying TfLite default delegates (i.e. the
      // XNNPACK delegate).
      const auto specified_delegates =
          tflite::KernelTestDelegateProviders::Get()->CreateAllDelegates();
      if (!specified_delegates.empty()) {
        bypass_default_delegates_ = true;
      }
    }
    MutableOpResolver* resolver =
        (bypass_default_delegates_ || use_simple_allocator)
            ? new ops::builtin::BuiltinOpResolverWithoutDefaultDelegates()
            : new ops::builtin::BuiltinOpResolver();
    for (const auto& reg : custom_registrations_) {
      resolver->AddCustom(reg.first.data(), reg.second());
    }
    resolver_ = std::unique_ptr<OpResolver>(resolver);
  }
  CHECK(InterpreterBuilder(GetModel(buffer_pointer), *resolver_)(
            &interpreter_, num_threads) == kTfLiteOk);

  CHECK(interpreter_ != nullptr);

  if (use_simple_allocator) {
    LOG(INFO) << "Use SimplePlanner.\n";
    tflite::Subgraph& primary_subgraph = interpreter_->primary_subgraph();
    auto memory_planner = new SimplePlanner(
        &primary_subgraph.context_,
        std::unique_ptr<GraphInfo>(primary_subgraph.CreateGraphInfo()));
    primary_subgraph.memory_planner_.reset(memory_planner);
    memory_planner->PlanAllocations();
  }

  for (size_t i = 0; i < input_shapes.size(); ++i) {
    const int input_idx = interpreter_->inputs()[i];
    if (input_idx == kTfLiteOptionalTensor) continue;
    const auto& shape = input_shapes[i];
    if (shape.empty()) continue;
    CHECK(interpreter_->ResizeInputTensor(input_idx, shape) == kTfLiteOk);
  }

  interpreter_->SetAllowFp16PrecisionForFp32(allow_fp32_relax_to_fp16);

  if (allocate_and_delegate) {
    AllocateAndDelegate(apply_delegate);
  }
}

TfLiteStatus SingleOpModel::ApplyDelegate() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_9(mht_9_v, 457, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::ApplyDelegate");

  if (delegate_) {
    TFLITE_LOG(WARN) << "Having a manually-set TfLite delegate, and bypassing "
                        "KernelTestDelegateProviders";
    TF_LITE_ENSURE_STATUS(interpreter_->ModifyGraphWithDelegate(delegate_));
    ++num_applied_delegates_;
  } else {
    auto* delegate_providers = tflite::KernelTestDelegateProviders::Get();
    // Most TFLite NNAPI delegation tests have been written to run against the
    // NNAPI CPU path. We'll enable that for tests. However, need to first check
    // if the parameter is present - it will not be if the NNAPI delegate
    // provider is not linked into the test.
    if (delegate_providers->ConstParams().HasParam("disable_nnapi_cpu")) {
      delegate_providers->MutableParams()->Set("disable_nnapi_cpu", false);
    }
    for (auto& one : delegate_providers->CreateAllDelegates()) {
      // The raw ptr always points to the actual TfLiteDegate object.
      auto* delegate_raw_ptr = one.delegate.get();
      TF_LITE_ENSURE_STATUS(
          interpreter_->ModifyGraphWithDelegate(std::move(one.delegate)));
      // Note: 'delegate_' is always set to the last successfully applied one.
      delegate_ = delegate_raw_ptr;
      ++num_applied_delegates_;
    }
  }
  return kTfLiteOk;
}

void SingleOpModel::Invoke() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_10(mht_10_v, 488, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::Invoke");
 ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk); }

TfLiteStatus SingleOpModel::InvokeUnchecked() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_11(mht_11_v, 493, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::InvokeUnchecked");
 return interpreter_->Invoke(); }

void SingleOpModel::BuildInterpreter(
    std::vector<std::vector<int>> input_shapes) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_12(mht_12_v, 499, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::BuildInterpreter");

  BuildInterpreter(input_shapes, /*num_threads=*/-1,
                   /*allow_fp32_relax_to_fp16=*/false,
                   /*apply_delegate=*/true, /*allocate_and_delegate=*/true);
}

// static
bool SingleOpModel::GetForceUseNnapi() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_13(mht_13_v, 509, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::GetForceUseNnapi");

  const auto& delegate_params =
      tflite::KernelTestDelegateProviders::Get()->ConstParams();
  // It's possible this library isn't linked with the nnapi delegate provider
  // lib.
  return delegate_params.HasParam("use_nnapi") &&
         delegate_params.Get<bool>("use_nnapi");
}

int32_t SingleOpModel::GetTensorSize(int index) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_14(mht_14_v, 521, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::GetTensorSize");

  TfLiteTensor* t = interpreter_->tensor(index);
  CHECK(t);
  int total_size = 1;
  for (int i = 0; i < t->dims->size; ++i) {
    total_size *= t->dims->data[i];
  }
  return total_size;
}

template <>
std::vector<string> SingleOpModel::ExtractVector(int index) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_15(mht_15_v, 535, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::ExtractVector");

  TfLiteTensor* tensor_ptr = interpreter_->tensor(index);
  CHECK(tensor_ptr != nullptr);
  const int num_strings = GetStringCount(tensor_ptr);
  std::vector<string> result;
  result.reserve(num_strings);
  for (int i = 0; i < num_strings; ++i) {
    const auto str = GetString(tensor_ptr, i);
    result.emplace_back(str.str, str.len);
  }
  return result;
}

namespace {

// Returns the number of partitions associated, as result of a call to
// ModifyGraphWithDelegate, to the given delegate.
int CountPartitionsDelegatedTo(Subgraph* subgraph,
                               const TfLiteDelegate* delegate) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_16(mht_16_v, 556, "", "./tensorflow/lite/kernels/test_util.cc", "CountPartitionsDelegatedTo");

  return std::count_if(
      subgraph->nodes_and_registration().begin(),
      subgraph->nodes_and_registration().end(),
      [delegate](
          std::pair<TfLiteNode, TfLiteRegistration> node_and_registration) {
        return node_and_registration.first.delegate == delegate;
      });
}

// Returns the number of partitions associated, as result of a call to
// ModifyGraphWithDelegate, to the given delegate.
int CountPartitionsDelegatedTo(Interpreter* interpreter,
                               const TfLiteDelegate* delegate) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_17(mht_17_v, 572, "", "./tensorflow/lite/kernels/test_util.cc", "CountPartitionsDelegatedTo");

  int result = 0;
  for (int i = 0; i < interpreter->subgraphs_size(); i++) {
    Subgraph* subgraph = interpreter->subgraph(i);

    result += CountPartitionsDelegatedTo(subgraph, delegate);
  }

  return result;
}

// Returns the number of nodes that will be executed on the CPU
int CountPartitionsExecutedByCpuKernel(const Interpreter* interpreter) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_18(mht_18_v, 587, "", "./tensorflow/lite/kernels/test_util.cc", "CountPartitionsExecutedByCpuKernel");

  int result = 0;
  for (int node_idx : interpreter->execution_plan()) {
    TfLiteNode node;
    TfLiteRegistration reg;
    std::tie(node, reg) = *(interpreter->node_and_registration(node_idx));

    if (node.delegate == nullptr) {
      ++result;
    }
  }

  return result;
}

}  // namespace

void SingleOpModel::ExpectOpAcceleratedWithNnapi(const std::string& test_id) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("test_id: \"" + test_id + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_19(mht_19_v, 608, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::ExpectOpAcceleratedWithNnapi");

  absl::optional<NnapiAccelerationTestParams> validation_params =
      GetNnapiAccelerationTestParam(test_id);
  if (!validation_params.has_value()) {
    return;
  }

  // If we have multiple delegates applied, we would skip this check at the
  // moment.
  if (num_applied_delegates_ > 1) {
    TFLITE_LOG(WARN) << "Skipping ExpectOpAcceleratedWithNnapi as "
                     << num_applied_delegates_
                     << " delegates have been successfully applied.";
    return;
  }
  TFLITE_LOG(INFO) << "Validating acceleration";
  const NnApi* nnapi = NnApiImplementation();
  if (nnapi && nnapi->nnapi_exists &&
      nnapi->android_sdk_version >=
          validation_params.value().MinAndroidSdkVersion()) {
    EXPECT_EQ(CountPartitionsDelegatedTo(interpreter_.get(), delegate_), 1)
        << "Expecting operation to be accelerated but cannot find a partition "
           "associated to the NNAPI delegate";
    EXPECT_GT(num_applied_delegates_, 0) << "No delegates were applied.";
  }
}

void SingleOpModel::ValidateAcceleration() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_20(mht_20_v, 638, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::ValidateAcceleration");

  if (GetForceUseNnapi()) {
    ExpectOpAcceleratedWithNnapi(GetCurrentTestId());
  }
}

int SingleOpModel::CountOpsExecutedByCpuKernel() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_21(mht_21_v, 647, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::CountOpsExecutedByCpuKernel");

  return CountPartitionsExecutedByCpuKernel(interpreter_.get());
}

SingleOpModel::~SingleOpModel() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_22(mht_22_v, 654, "", "./tensorflow/lite/kernels/test_util.cc", "SingleOpModel::~SingleOpModel");
 ValidateAcceleration(); }

void MultiOpModel::AddBuiltinOp(
    BuiltinOperator type, BuiltinOptions builtin_options_type,
    const flatbuffers::Offset<void>& builtin_options,
    const std::vector<int32_t>& inputs, const std::vector<int32_t>& outputs) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_23(mht_23_v, 662, "", "./tensorflow/lite/kernels/test_util.cc", "MultiOpModel::AddBuiltinOp");

  opcodes_.push_back(CreateOperatorCode(builder_, type, 0, 0));
  const int opcode_index = opcodes_.size() - 1;
  operators_.push_back(CreateOperator(
      builder_, opcode_index, builder_.CreateVector<int32_t>(inputs),
      builder_.CreateVector<int32_t>(outputs), builtin_options_type,
      builtin_options,
      /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS));
}

void MultiOpModel::AddCustomOp(
    const string& name, const std::vector<uint8_t>& custom_option,
    const std::function<TfLiteRegistration*()>& registration,
    const std::vector<int32_t>& inputs, const std::vector<int32_t>& outputs) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPSlitePSkernelsPStest_utilDTcc mht_24(mht_24_v, 679, "", "./tensorflow/lite/kernels/test_util.cc", "MultiOpModel::AddCustomOp");

  custom_registrations_[name] = registration;
  opcodes_.push_back(
      CreateOperatorCodeDirect(builder_, BuiltinOperator_CUSTOM, name.data()));
  const int opcode_index = opcodes_.size() - 1;
  operators_.push_back(CreateOperator(
      builder_, opcode_index, builder_.CreateVector<int32_t>(inputs),
      builder_.CreateVector<int32_t>(outputs), BuiltinOptions_NONE, 0,
      builder_.CreateVector<uint8_t>(custom_option),
      CustomOptionsFormat_FLEXBUFFERS));
}
}  // namespace tflite
