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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/flex/test_util.h"

#include "absl/memory/memory.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace flex {
namespace testing {

bool FlexModelTest::Invoke() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::Invoke");
 return interpreter_->Invoke() == kTfLiteOk; }

void FlexModelTest::SetStringValues(int tensor_index,
                                    const std::vector<string>& values) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_1(mht_1_v, 201, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::SetStringValues");

  DynamicBuffer dynamic_buffer;
  for (const string& s : values) {
    dynamic_buffer.AddString(s.data(), s.size());
  }
  dynamic_buffer.WriteToTensor(interpreter_->tensor(tensor_index),
                               /*new_shape=*/nullptr);
}

std::vector<string> FlexModelTest::GetStringValues(int tensor_index) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_2(mht_2_v, 213, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::GetStringValues");

  std::vector<string> result;

  TfLiteTensor* tensor = interpreter_->tensor(tensor_index);
  auto num_strings = GetStringCount(tensor);
  for (size_t i = 0; i < num_strings; ++i) {
    auto ref = GetString(tensor, i);
    result.push_back(string(ref.str, ref.len));
  }

  return result;
}

void FlexModelTest::SetShape(int tensor_index, const std::vector<int>& values) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_3(mht_3_v, 229, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::SetShape");

  ASSERT_EQ(interpreter_->ResizeInputTensor(tensor_index, values), kTfLiteOk);
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
}

std::vector<int> FlexModelTest::GetShape(int tensor_index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_4(mht_4_v, 237, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::GetShape");

  std::vector<int> result;
  auto* dims = interpreter_->tensor(tensor_index)->dims;
  result.reserve(dims->size);
  for (int i = 0; i < dims->size; ++i) {
    result.push_back(dims->data[i]);
  }
  return result;
}

TfLiteType FlexModelTest::GetType(int tensor_index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_5(mht_5_v, 250, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::GetType");

  return interpreter_->tensor(tensor_index)->type;
}

bool FlexModelTest::IsDynamicTensor(int tensor_index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_6(mht_6_v, 257, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::IsDynamicTensor");

  return interpreter_->tensor(tensor_index)->allocation_type == kTfLiteDynamic;
}

void FlexModelTest::AddTensors(int num_tensors, const std::vector<int>& inputs,
                               const std::vector<int>& outputs, TfLiteType type,
                               const std::vector<int>& dims) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_7(mht_7_v, 266, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::AddTensors");

  interpreter_->AddTensors(num_tensors);
  for (int i = 0; i < num_tensors; ++i) {
    TfLiteQuantizationParams quant;
    CHECK_EQ(interpreter_->SetTensorParametersReadWrite(i, type,
                                                        /*name=*/"",
                                                        /*dims=*/dims, quant),
             kTfLiteOk);
  }

  CHECK_EQ(interpreter_->SetInputs(inputs), kTfLiteOk);
  CHECK_EQ(interpreter_->SetOutputs(outputs), kTfLiteOk);
}

void FlexModelTest::SetConstTensor(int tensor_index,
                                   const std::vector<int>& values,
                                   TfLiteType type, const char* buffer,
                                   size_t bytes) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("buffer: \"" + (buffer == nullptr ? std::string("nullptr") : std::string((char*)buffer)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_8(mht_8_v, 287, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::SetConstTensor");

  TfLiteQuantizationParams quant;
  CHECK_EQ(interpreter_->SetTensorParametersReadOnly(tensor_index, type,
                                                     /*name=*/"",
                                                     /*dims=*/values, quant,
                                                     buffer, bytes),
           kTfLiteOk);
}

void FlexModelTest::AddTfLiteMulOp(const std::vector<int>& inputs,
                                   const std::vector<int>& outputs) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_9(mht_9_v, 300, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::AddTfLiteMulOp");

  ++next_op_index_;

  static TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  reg.builtin_code = BuiltinOperator_MUL;
  reg.prepare = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_10(mht_10_v, 308, "", "./tensorflow/lite/delegates/flex/test_util.cc", "lambda");

    auto* i0 = &context->tensors[node->inputs->data[0]];
    auto* o = &context->tensors[node->outputs->data[0]];
    return context->ResizeTensor(context, o, TfLiteIntArrayCopy(i0->dims));
  };
  reg.invoke = [](TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_11(mht_11_v, 316, "", "./tensorflow/lite/delegates/flex/test_util.cc", "lambda");

    auto* i0 = &context->tensors[node->inputs->data[0]];
    auto* i1 = &context->tensors[node->inputs->data[1]];
    auto* o = &context->tensors[node->outputs->data[0]];
    for (int i = 0; i < o->bytes / sizeof(float); ++i) {
      o->data.f[i] = i0->data.f[i] * i1->data.f[i];
    }
    return kTfLiteOk;
  };

  CHECK_EQ(interpreter_->AddNodeWithParameters(inputs, outputs, nullptr, 0,
                                               nullptr, &reg),
           kTfLiteOk);
}

void FlexModelTest::AddTfOp(TfOpType op, const std::vector<int>& inputs,
                            const std::vector<int>& outputs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_12(mht_12_v, 335, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::AddTfOp");

  tf_ops_.push_back(next_op_index_);
  ++next_op_index_;

  auto attr = [](const string& key, const string& value) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("key: \"" + key + "\"");
   mht_13_v.push_back("value: \"" + value + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_13(mht_13_v, 344, "", "./tensorflow/lite/delegates/flex/test_util.cc", "lambda");

    return " attr{ key: '" + key + "' value {" + value + "}}";
  };

  string type_attribute;
  switch (interpreter_->tensor(inputs[0])->type) {
    case kTfLiteInt32:
      type_attribute = attr("T", "type: DT_INT32");
      break;
    case kTfLiteFloat32:
      type_attribute = attr("T", "type: DT_FLOAT");
      break;
    case kTfLiteString:
      type_attribute = attr("T", "type: DT_STRING");
      break;
    case kTfLiteBool:
      type_attribute = attr("T", "type: DT_BOOL");
      break;
    default:
      // TODO(b/113613439): Use nodedef string utilities to properly handle all
      // types.
      LOG(FATAL) << "Type not supported";
      break;
  }

  if (op == kUnpack) {
    string attributes =
        type_attribute + attr("num", "i: 2") + attr("axis", "i: 0");
    AddTfOp("FlexUnpack", "Unpack", attributes, inputs, outputs);
  } else if (op == kIdentity) {
    string attributes = type_attribute;
    AddTfOp("FlexIdentity", "Identity", attributes, inputs, outputs);
  } else if (op == kAdd) {
    string attributes = type_attribute;
    AddTfOp("FlexAdd", "Add", attributes, inputs, outputs);
  } else if (op == kMul) {
    string attributes = type_attribute;
    AddTfOp("FlexMul", "Mul", attributes, inputs, outputs);
  } else if (op == kRfft) {
    AddTfOp("FlexRFFT", "RFFT", "", inputs, outputs);
  } else if (op == kImag) {
    AddTfOp("FlexImag", "Imag", "", inputs, outputs);
  } else if (op == kLoopCond) {
    string attributes = type_attribute;
    AddTfOp("FlexLoopCond", "LoopCond", attributes, inputs, outputs);
  } else if (op == kNonExistent) {
    AddTfOp("NonExistentOp", "NonExistentOp", "", inputs, outputs);
  } else if (op == kIncompatibleNodeDef) {
    // "Cast" op is created without attributes - making it incompatible.
    AddTfOp("FlexCast", "Cast", "", inputs, outputs);
  }
}

void FlexModelTest::AddTfOp(const char* tflite_name, const string& tf_name,
                            const string& nodedef_str,
                            const std::vector<int>& inputs,
                            const std::vector<int>& outputs) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("tflite_name: \"" + (tflite_name == nullptr ? std::string("nullptr") : std::string((char*)tflite_name)) + "\"");
   mht_14_v.push_back("tf_name: \"" + tf_name + "\"");
   mht_14_v.push_back("nodedef_str: \"" + nodedef_str + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSflexPStest_utilDTcc mht_14(mht_14_v, 406, "", "./tensorflow/lite/delegates/flex/test_util.cc", "FlexModelTest::AddTfOp");

  static TfLiteRegistration reg = {nullptr, nullptr, nullptr, nullptr};
  reg.builtin_code = BuiltinOperator_CUSTOM;
  reg.custom_name = tflite_name;

  tensorflow::NodeDef nodedef;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      nodedef_str + " op: '" + tf_name + "'", &nodedef));
  string serialized_nodedef;
  CHECK(nodedef.SerializeToString(&serialized_nodedef));
  flexbuffers::Builder fbb;
  fbb.Vector([&]() {
    fbb.String(nodedef.op());
    fbb.String(serialized_nodedef);
  });
  fbb.Finish();

  flexbuffers_.push_back(fbb.GetBuffer());
  auto& buffer = flexbuffers_.back();
  CHECK_EQ(interpreter_->AddNodeWithParameters(
               inputs, outputs, reinterpret_cast<const char*>(buffer.data()),
               buffer.size(), nullptr, &reg),
           kTfLiteOk);
}

}  // namespace testing
}  // namespace flex
}  // namespace tflite
