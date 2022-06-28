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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc() {
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
#include "tensorflow/core/common_runtime/eager/eager_operation.h"

#include "absl/types/span.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/custom_device.h"
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/host_info.h"

namespace tensorflow {

// An EagerOperation object can be reused for a different op by calling
// Clear(), and then Reset(...) with the same arguments that would have
// been provided to the constructor.
void EagerOperation::Clear() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::Clear");

  for (ImmediateExecutionTensorHandle* h : inputs_) {
    h->Unref();
  }
  inputs_.clear();
  custom_device_tensor_handles_count_ = 0;
  ClearInferenceState();
}

Status EagerOperation::SetAttrValue(const char* attr_name,
                                    const AttrValue& value) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrValue");

  MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status EagerOperation::SetAttrString(const char* attr_name, const char* data,
                                     size_t length) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_2_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_2(mht_2_v, 228, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrString");

  MutableAttrs()->Set(attr_name, StringPiece(data, length));
  return Status::OK();
}

Status EagerOperation::SetAttrInt(const char* attr_name, int64_t value) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_3(mht_3_v, 237, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrInt");

  MutableAttrs()->Set(attr_name, static_cast<int64_t>(value));
  return Status::OK();
}

Status EagerOperation::SetAttrFloat(const char* attr_name, float value) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_4(mht_4_v, 246, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrFloat");

  MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status EagerOperation::SetAttrBool(const char* attr_name, bool value) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_5(mht_5_v, 255, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrBool");

  MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status EagerOperation::SetAttrType(const char* attr_name, DataType value) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_6(mht_6_v, 264, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrType");

  MutableAttrs()->Set(attr_name, value);
  return Status::OK();
}

Status EagerOperation::SetAttrShape(const char* attr_name, const int64_t* dims,
                                    const int num_dims) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_7(mht_7_v, 274, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrShape");

  if (num_dims > TensorShape::MaxDimensions()) {
    return errors::InvalidArgument("Value specified for `", attr_name, "` has ",
                                   num_dims,
                                   " dimensions which is over the limit of ",
                                   TensorShape::MaxDimensions(), ".");
  }

  TensorShapeProto proto;
  if (num_dims < 0) {
    proto.set_unknown_rank(true);
  } else {
    for (int d = 0; d < num_dims; ++d) {
      proto.add_dim()->set_size(dims[d]);
    }
  }

  MutableAttrs()->Set(attr_name, proto);

  return Status::OK();
}

Status EagerOperation::SetAttrFunction(const char* attr_name,
                                       const AbstractOperation* value) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_8(mht_8_v, 301, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrFunction");

  AttrValue attr_value;
  NameAttrList* func = attr_value.mutable_func();
  func->set_name(value->Name());
  auto* value_operation = down_cast<const EagerOperation*>(value);
  value_operation->Attrs().FillAttrValueMap(func->mutable_attr());
  MutableAttrs()->Set(attr_name, attr_value);
  return Status::OK();
}

Status EagerOperation::SetAttrFunctionName(const char* attr_name,
                                           const char* data, size_t length) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_9_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_9(mht_9_v, 317, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrFunctionName");

  AttrValue attr_value;
  NameAttrList* func = attr_value.mutable_func();
  func->set_name(data, length);
  MutableAttrs()->Set(attr_name, attr_value);
  return Status::OK();
}

Status EagerOperation::SetAttrTensor(const char* attr_name,
                                     AbstractTensorInterface* tensor) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_10(mht_10_v, 330, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrTensor");

  Tensor t = TensorFromInterface(tensor);
  MutableAttrs()->Set(attr_name, t);
  return Status::OK();
}

Status EagerOperation::SetAttrStringList(const char* attr_name,
                                         const void* const* values,
                                         const size_t* lengths,
                                         int num_values) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_11(mht_11_v, 343, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrStringList");

  std::vector<StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = StringPiece(static_cast<const char*>(values[i]), lengths[i]);
  }
  MutableAttrs()->Set(attr_name, v);

  return Status::OK();
}

Status EagerOperation::SetAttrFloatList(const char* attr_name,
                                        const float* values, int num_values) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_12(mht_12_v, 358, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrFloatList");

  MutableAttrs()->Set(attr_name,
                      gtl::ArraySlice<const float>(values, num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrIntList(const char* attr_name,
                                      const int64_t* values, int num_values) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_13(mht_13_v, 369, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrIntList");

  MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<const int64_t>(
                     reinterpret_cast<const int64_t*>(values), num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrTypeList(const char* attr_name,
                                       const DataType* values, int num_values) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_14(mht_14_v, 381, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrTypeList");

  MutableAttrs()->Set(attr_name,
                      gtl::ArraySlice<const DataType>(values, num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrBoolList(const char* attr_name,
                                       const unsigned char* values,
                                       int num_values) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_15_v.push_back("values: \"" + (values == nullptr ? std::string("nullptr") : std::string((char*)values)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_15(mht_15_v, 394, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrBoolList");

  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  MutableAttrs()->Set(attr_name,
                      gtl::ArraySlice<const bool>(b.get(), num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrShapeList(const char* attr_name,
                                        const int64_t** dims,
                                        const int* num_dims, int num_values) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_16(mht_16_v, 410, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrShapeList");

  std::unique_ptr<TensorShapeProto[]> proto(new TensorShapeProto[num_values]);
  for (int i = 0; i < num_values; ++i) {
    const auto num_dims_i = num_dims[i];

    if (num_dims_i > TensorShape::MaxDimensions()) {
      return errors::InvalidArgument(
          strings::StrCat("Value specified for `", attr_name, "` has ",
                          num_dims_i, " dimensions which is over the limit of ",
                          TensorShape::MaxDimensions(), "."));
    }
    if (num_dims_i < 0) {
      proto[i].set_unknown_rank(true);
    } else {
      const int64_t* dims_i = dims[i];
      auto proto_i = &proto[i];
      for (int d = 0; d < num_dims_i; ++d) {
        proto_i->add_dim()->set_size(dims_i[d]);
      }
    }
  }
  MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<TensorShapeProto>(proto.get(), num_values));
  return Status::OK();
}

Status EagerOperation::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_17(mht_17_v, 441, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetAttrFunctionList");

  size_t num_values = values.size();
  std::unique_ptr<NameAttrList[]> funcs(new NameAttrList[num_values]);
  for (int i = 0; i < num_values; i++) {
    auto* value_operation = down_cast<const EagerOperation*>(values[i]);
    funcs[i].set_name(value_operation->Name());
    value_operation->Attrs().FillAttrValueMap(funcs[i].mutable_attr());
  }
  MutableAttrs()->Set(
      attr_name, gtl::ArraySlice<const NameAttrList>(funcs.get(), num_values));
  return Status::OK();
}

const OpDef* EagerOperation::GetOpDef(Status* status) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_18(mht_18_v, 457, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::GetOpDef");

  const tensorflow::OpDef* op_def = OpDef();
  if (op_def) return op_def;
  *status = OpDefForOp(Name(), &op_def);
  return op_def;
}

Status EagerOperation::InputLength(const char* input_name, int* length) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("input_name: \"" + (input_name == nullptr ? std::string("nullptr") : std::string((char*)input_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_19(mht_19_v, 468, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::InputLength");

  Status status;
  const tensorflow::OpDef* op_def = GetOpDef(&status);
  if (!status.ok()) {
    return status;
  }
  AttrValueMap attrs;
  Attrs().FillAttrValueMap(&attrs);
  NameRangeMap name_ranges;
  TF_RETURN_IF_ERROR(
      NameRangesForNode(AttrSlice(&attrs), *op_def, &name_ranges, nullptr));
  auto iter = name_ranges.find(input_name);
  if (iter == name_ranges.end()) {
    return errors::InvalidArgument("Input '", input_name, "' not found");
  }
  *length = iter->second.second - iter->second.first;
  return Status::OK();
}

absl::Span<ImmediateExecutionTensorHandle* const> EagerOperation::GetInputs()
    const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_20(mht_20_v, 491, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::GetInputs");

  // TODO(b/162536003): Remove reinterpret_cast.
  return absl::MakeSpan(
      reinterpret_cast<ImmediateExecutionTensorHandle* const*>(inputs_.data()),
      inputs_.size());
}

Status EagerOperation::OutputLength(const char* output_name, int* length) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("output_name: \"" + (output_name == nullptr ? std::string("nullptr") : std::string((char*)output_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_21(mht_21_v, 502, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::OutputLength");

  Status status;
  const tensorflow::OpDef* op_def = GetOpDef(&status);
  if (!status.ok()) {
    return status;
  }
  AttrValueMap attrs;
  Attrs().FillAttrValueMap(&attrs);
  NameRangeMap name_ranges;
  TF_RETURN_IF_ERROR(
      NameRangesForNode(AttrSlice(&attrs), *op_def, nullptr, &name_ranges));
  auto iter = name_ranges.find(output_name);
  if (iter == name_ranges.end()) {
    return errors::InvalidArgument("Output '", output_name, "' not found");
  }
  *length = iter->second.second - iter->second.first;
  return Status::OK();
}

Status EagerOperation::AddInput(AbstractTensorHandle* input) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_22(mht_22_v, 524, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::AddInput");

  ImmediateExecutionTensorHandle* h =
      down_cast<ImmediateExecutionTensorHandle*>(input);
  // TODO(b/175427838): It would be nice to be able to use tensorflow::isa here.
  if (CustomDeviceTensorHandle::classof(h)) {
    custom_device_tensor_handles_count_++;
  }
  AddTensorHandle(h);
  return MaybeInferSingleInputAttrs(h);
}

Status EagerOperation::AddInputList(
    absl::Span<AbstractTensorHandle* const> inputs) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_23(mht_23_v, 539, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::AddInputList");

  for (auto& input : inputs) {
    // TODO(b/175427838): It would be nice to be able to use tensorflow::isa
    // here.
    if (CustomDeviceTensorHandle::classof(input)) {
      custom_device_tensor_handles_count_++;
    }
    ImmediateExecutionTensorHandle* h =
        down_cast<ImmediateExecutionTensorHandle*>(input);
    AddTensorHandle(h);
  }
  return InferInputListAttrs(inputs.size());
}

Status EagerOperation::SetInput(size_t index,
                                ImmediateExecutionTensorHandle* input) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_24(mht_24_v, 557, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetInput");

  if (index >= inputs_.size()) {
    return errors::InvalidArgument("Index >= inputs.size: %d >= %d", index,
                                   inputs_.size());
  }
  auto* previous = inputs_[index];
  if (CustomDeviceTensorHandle::classof(previous)) {
    custom_device_tensor_handles_count_--;
  }
  if (CustomDeviceTensorHandle::classof(input)) {
    custom_device_tensor_handles_count_++;
  }
  input->Ref();
  inputs_[index] = input;
  previous->Unref();
  return Status::OK();
}

Status EagerOperation::Reset(
    const char* op, const char* device_name, bool remote,
    EagerExecutor* executor,
    const absl::optional<EagerFunctionParams> eager_func_params) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   mht_25_v.push_back("device_name: \"" + (device_name == nullptr ? std::string("nullptr") : std::string((char*)device_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_25(mht_25_v, 583, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::Reset");

  DCHECK(inputs_.empty());
  ClearInferenceState();
  bool is_function = false;
  TF_RETURN_IF_ERROR(AttrTypeMapForOp(op, &attr_types_, &is_function));

  // Don't update the device of direct function calls.
  // Particularly, if the user did not explicitly request any device for this
  // function, picking a device would result in this device being the default
  // for nodes inside the function. This is undesirable for multi-device
  // functions since the not-explicitly-placed nodes inside the body will all
  // end up on this default device.
  colocation_exempt_ = is_function;
  if (!is_function) {
    const auto& exempt_ops = InputColocationExemptionRegistry::Global()->Get();
    colocation_exempt_ = exempt_ops.find(op) != exempt_ops.end();

    TF_RETURN_IF_ERROR(OpDefForOp(op, &op_def_));
  } else if (!remote && !ctx_.FindFunctionByName(op)) {
    return errors::NotFound(
        "'", op,
        "' is neither a type of a primitive operation nor a name "
        "of a function registered in binary running on ",
        port::Hostname(),
        ". Make sure the operation or function is "
        "registered in the binary running in this process.");
  }
  attrs_.Reset(op);
  stack_trace_.reset();
  is_function_ = is_function;
  cancellation_manager_ = nullptr;
  executor_ = executor ? executor : &ctx_.Executor();
  if (eager_func_params.has_value()) {
    eager_func_params_ = eager_func_params;
  }
  op_name_ = op;
  return SetDeviceName(device_name);
}

Status EagerOperation::MaybeInferSingleInputAttrs(
    ImmediateExecutionTensorHandle* handle) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_26(mht_26_v, 626, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::MaybeInferSingleInputAttrs");

  if (!op_def_) return Status::OK();

  const auto& input_def = op_def_->input_arg(inference_arg_idx_++);
  if (!input_def.number_attr().empty() || !input_def.type_list_attr().empty()) {
    // Some clients that are still setting their input attributes manually are
    // adding input list to their op by calling `TFE_OpAddInput` for each of
    // its elements instead of calling `TFE_OpAddInputList`. When this happens,
    // we cannot detect the end of such list, thus lose track of the input
    // arguments in the op definition. To guarantee backward compatibility with
    // those clients, disable automatic inference in this case.
    ClearInferenceState();
    return Status::OK();
  }
  const std::string& type_attr = input_def.type_attr();
  if (!type_attr.empty() &&
      inference_attrs_.find(type_attr) == inference_attrs_.end()) {
    MutableAttrs()->Set(type_attr, handle->DataType());
    inference_attrs_.insert(type_attr);
  }
  return Status::OK();
}

void EagerOperation::InferSingleTypeInputListAttrs(
    const OpDef::ArgDef& input_def, const DataType dtype, int num_inputs) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_27(mht_27_v, 653, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::InferSingleTypeInputListAttrs");

  if (inference_attrs_.find(input_def.number_attr()) ==
      inference_attrs_.end()) {
    MutableAttrs()->Set(input_def.number_attr(), num_inputs);
    inference_attrs_.insert(input_def.number_attr());
  }
  if (inference_attrs_.find(input_def.type_attr()) == inference_attrs_.end()) {
    MutableAttrs()->Set(input_def.type_attr(), dtype);
    inference_attrs_.insert(input_def.type_attr());
  }
}

void EagerOperation::InferMixedTypeInputListAttrs(
    const OpDef::ArgDef& input_def, const std::vector<DataType>& dtypes) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_28(mht_28_v, 669, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::InferMixedTypeInputListAttrs");

  if (inference_attrs_.find(input_def.type_list_attr()) ==
      inference_attrs_.end()) {
    MutableAttrs()->Set(
        input_def.type_list_attr(),
        gtl::ArraySlice<const DataType>(dtypes.data(), dtypes.size()));
    inference_attrs_.insert(input_def.type_list_attr());
  }
}

Status EagerOperation::InferInputListAttrs(int num_inputs) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_29(mht_29_v, 682, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::InferInputListAttrs");

  if (!op_def_) return Status::OK();

  int start = inference_arg_idx_;
  const auto& input_def = op_def_->input_arg(inference_arg_idx_++);
  if (!input_def.type_list_attr().empty()) {
    std::vector<DataType> dtypes(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      dtypes[i] = inputs_[start + i]->DataType();
    }
    InferMixedTypeInputListAttrs(input_def, dtypes);
  } else if (!input_def.type_attr().empty() &&
             !input_def.number_attr().empty()) {
    InferSingleTypeInputListAttrs(input_def, inputs_[start]->DataType(),
                                  num_inputs);
  } else if (!input_def.number_attr().empty()) {
    if (inference_attrs_.find(input_def.number_attr()) ==
        inference_attrs_.end()) {
      MutableAttrs()->Set(input_def.number_attr(), num_inputs);
      inference_attrs_.insert(input_def.number_attr());
    }
  } else {
    return errors::InvalidArgument("Invalid input list definition");
  }
  return Status::OK();
}

Status EagerOperation::TensorHandleInputs(
    const absl::InlinedVector<TensorHandle*, 4>** inputs) const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_30(mht_30_v, 713, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::TensorHandleInputs");

  if (TF_PREDICT_TRUE(!HasCustomDeviceInput())) {
    *inputs = reinterpret_cast<const absl::InlinedVector<TensorHandle*, 4>*>(
        &inputs_);
    return Status::OK();
  } else {
    return errors::Internal("The operation unexpectedly had custom devices.");
  }
}

Status EagerOperation::MutableTensorHandleInputs(
    absl::InlinedVector<TensorHandle*, 4>** inputs) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_31(mht_31_v, 727, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::MutableTensorHandleInputs");

  if (TF_PREDICT_TRUE(!HasCustomDeviceInput())) {
    *inputs =
        reinterpret_cast<absl::InlinedVector<TensorHandle*, 4>*>(&inputs_);
    return Status::OK();
  } else {
    return errors::Internal("The operation unexpectedly had custom devices.");
  }
}

Status EagerOperation::SetDeviceName(const char* c_name) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("c_name: \"" + (c_name == nullptr ? std::string("nullptr") : std::string((char*)c_name)) + "\"");
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_32(mht_32_v, 741, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::SetDeviceName");

  string name(c_name != nullptr ? c_name : "");
  if (name != last_set_device_name_) {
    if (!DeviceNameUtils::ParseFullName(name, &device_parsed_name_)) {
      return errors::InvalidArgument("Malformed device specification '", name,
                                     "' in eager op: ", DebugString());
    }
    last_set_device_name_ = name;
    device_name_ = DeviceNameUtils::ParsedNameToString(device_parsed_name_);
    device_ = kVariantDeviceNull;
  }
  return Status::OK();
}

bool EagerOperation::IsLocal() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_33(mht_33_v, 758, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::IsLocal");

  if (ctx_.remote_device_mgr() == nullptr) return true;

  if (!device_parsed_name_.has_job && !device_parsed_name_.has_replica &&
      !device_parsed_name_.has_task)
    return true;
  auto& host_cpu_name = ctx_.HostCPU()->parsed_name();
  return device_parsed_name_.job == host_cpu_name.job &&
         device_parsed_name_.replica == host_cpu_name.replica &&
         device_parsed_name_.task == host_cpu_name.task;
}

string VariantDeviceDebugString(VariantDevice device) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_34(mht_34_v, 773, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "VariantDeviceDebugString");

  if (device == kVariantDeviceNull) {
    return "[]";
  } else if (absl::holds_alternative<CustomDevice*>(device)) {
    return absl::get<CustomDevice*>(device)->name();
  } else {
    return absl::get<Device*>(device)->DebugString();
  }
}
const AbstractOpAttrs* EagerOperation::GetOpAttrs() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_35(mht_35_v, 785, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::GetOpAttrs");
 return &attrs_; }

void EagerOperation::AddAttrs(const AbstractOpAttrs* op_attrs) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_36(mht_36_v, 790, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::AddAttrs");

  attrs_.CopyAttributes(*(down_cast<const AttrBuilder*>(op_attrs)));
}

string EagerOperation::DebugString() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_37(mht_37_v, 797, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::DebugString");

  string out;
  VLOG(1) << "EagerOperation::DebugString() over " << this;

  strings::StrAppend(&out, "Name: ", Name(), "\n");
  strings::StrAppend(&out, "Device Name: [", device_name_, "]\n");
  strings::StrAppend(&out, "Device: ", VariantDeviceDebugString(Device()),
                     "\n");
  for (const auto& input : inputs_) {
    VLOG(1) << "Input ptr: " << input;
    strings::StrAppend(&out, "Input: ", input->DebugString(), "\n");
  }

  NodeDef ndef;
  Attrs().FillAttrValueMap(ndef.mutable_attr());
  strings::StrAppend(&out, "Attrs: ", ndef.DebugString(), "\n");
  return out;
}

void EagerOperation::AddTensorHandle(ImmediateExecutionTensorHandle* h) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_operationDTcc mht_38(mht_38_v, 819, "", "./tensorflow/core/common_runtime/eager/eager_operation.cc", "EagerOperation::AddTensorHandle");

  h->Ref();
  inputs_.push_back(h);
  attrs_.NumInputs(static_cast<int>(inputs_.size()));
}
}  // namespace tensorflow
