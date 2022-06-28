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
class MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc() {
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
#include "tensorflow/c/experimental/gradients/tape/tape_operation.h"

#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/gradients.h"

namespace tensorflow {
namespace gradients {
TapeOperation::TapeOperation(AbstractOperation* parent_op, Tape* tape,
                             const GradientRegistry& registry)
    : AbstractOperation(kTape),
      parent_op_(parent_op),
      tape_(tape),
      registry_(registry) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_0(mht_0_v, 196, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::TapeOperation");

  // TODO(b/172003047): Consider making AbstractOperation RefCounted.
  // parent_op_->Ref();
}
void TapeOperation::Release() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_1(mht_1_v, 203, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::Release");

  // TODO(srbs): Change to Unref().
  delete this;
}
TapeOperation::~TapeOperation() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_2(mht_2_v, 210, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::~TapeOperation");

  // TODO(b/172003047): Consider making AbstractOperation RefCounted.
  // parent_op->Unref();
}
Status TapeOperation::Reset(const char* op, const char* raw_device_name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("op: \"" + (op == nullptr ? std::string("nullptr") : std::string((char*)op)) + "\"");
   mht_3_v.push_back("raw_device_name: \"" + (raw_device_name == nullptr ? std::string("nullptr") : std::string((char*)raw_device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_3(mht_3_v, 219, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::Reset");

  forward_op_.op_name = op;
  forward_op_.attrs.Reset(op);
  forward_op_.inputs.clear();
  forward_op_.outputs.clear();
  return parent_op_->Reset(op, raw_device_name);
}
const string& TapeOperation::Name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_4(mht_4_v, 229, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::Name");
 return parent_op_->Name(); }
const string& TapeOperation::DeviceName() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_5(mht_5_v, 233, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::DeviceName");

  return parent_op_->DeviceName();
}
Status TapeOperation::SetDeviceName(const char* name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_6(mht_6_v, 240, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetDeviceName");

  return parent_op_->SetDeviceName(name);
}
Status TapeOperation::AddInput(AbstractTensorHandle* input) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_7(mht_7_v, 246, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::AddInput");

  TF_RETURN_IF_ERROR(parent_op_->AddInput(input));
  forward_op_.inputs.push_back(input);
  return Status::OK();
}
Status TapeOperation::AddInputList(
    absl::Span<AbstractTensorHandle* const> inputs) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_8(mht_8_v, 255, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::AddInputList");

  TF_RETURN_IF_ERROR(parent_op_->AddInputList(inputs));
  for (auto input : inputs) {
    forward_op_.inputs.push_back(input);
  }
  return Status::OK();
}
Status TapeOperation::SetAttrString(const char* attr_name, const char* data,
                                    size_t length) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_9_v.push_back("data: \"" + (data == nullptr ? std::string("nullptr") : std::string((char*)data)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_9(mht_9_v, 268, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrString");

  forward_op_.attrs.Set(attr_name, StringPiece(data, length));
  return parent_op_->SetAttrString(attr_name, data, length);
}
Status TapeOperation::SetAttrInt(const char* attr_name, int64_t value) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_10(mht_10_v, 276, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrInt");

  forward_op_.attrs.Set(attr_name, static_cast<int64_t>(value));
  return parent_op_->SetAttrInt(attr_name, value);
}
Status TapeOperation::SetAttrFloat(const char* attr_name, float value) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_11(mht_11_v, 284, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrFloat");

  forward_op_.attrs.Set(attr_name, value);
  return parent_op_->SetAttrFloat(attr_name, value);
}
Status TapeOperation::SetAttrBool(const char* attr_name, bool value) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_12(mht_12_v, 292, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrBool");

  forward_op_.attrs.Set(attr_name, value);
  return parent_op_->SetAttrBool(attr_name, value);
}
Status TapeOperation::SetAttrType(const char* attr_name, DataType value) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_13(mht_13_v, 300, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrType");

  forward_op_.attrs.Set(attr_name, value);
  return parent_op_->SetAttrType(attr_name, value);
}
Status TapeOperation::SetAttrShape(const char* attr_name, const int64_t* dims,
                                   const int num_dims) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_14(mht_14_v, 309, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrShape");

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

  forward_op_.attrs.Set(attr_name, proto);
  return parent_op_->SetAttrShape(attr_name, dims, num_dims);
}
Status TapeOperation::SetAttrFunction(const char* attr_name,
                                      const AbstractOperation* value) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_15(mht_15_v, 333, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrFunction");

  return tensorflow::errors::Unimplemented(
      "SetAttrFunction has not been implemented yet.");
}
Status TapeOperation::SetAttrFunctionName(const char* attr_name,
                                          const char* value, size_t length) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_16_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_16(mht_16_v, 343, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrFunctionName");

  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionName has not been implemented "
      "yet.");
}
Status TapeOperation::SetAttrTensor(const char* attr_name,
                                    AbstractTensorInterface* tensor) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_17(mht_17_v, 353, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrTensor");

  return tensorflow::errors::Unimplemented(
      "SetAttrTensor has not been implemented yet.");
}
Status TapeOperation::SetAttrStringList(const char* attr_name,
                                        const void* const* values,
                                        const size_t* lengths, int num_values) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_18(mht_18_v, 363, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrStringList");

  std::vector<StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = StringPiece(static_cast<const char*>(values[i]), lengths[i]);
  }
  forward_op_.attrs.Set(attr_name, v);
  return parent_op_->SetAttrStringList(attr_name, values, lengths, num_values);
}
Status TapeOperation::SetAttrFloatList(const char* attr_name,
                                       const float* values, int num_values) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_19(mht_19_v, 376, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrFloatList");

  forward_op_.attrs.Set(attr_name,
                        gtl::ArraySlice<const float>(values, num_values));
  return parent_op_->SetAttrFloatList(attr_name, values, num_values);
}
Status TapeOperation::SetAttrIntList(const char* attr_name,
                                     const int64_t* values, int num_values) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_20(mht_20_v, 386, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrIntList");

  forward_op_.attrs.Set(
      attr_name, gtl::ArraySlice<const int64_t>(
                     reinterpret_cast<const int64_t*>(values), num_values));
  return parent_op_->SetAttrIntList(attr_name, values, num_values);
}
Status TapeOperation::SetAttrTypeList(const char* attr_name,
                                      const DataType* values, int num_values) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_21(mht_21_v, 397, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrTypeList");

  forward_op_.attrs.Set(attr_name,
                        gtl::ArraySlice<const DataType>(values, num_values));
  return parent_op_->SetAttrTypeList(attr_name, values, num_values);
}
Status TapeOperation::SetAttrBoolList(const char* attr_name,
                                      const unsigned char* values,
                                      int num_values) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   mht_22_v.push_back("values: \"" + (values == nullptr ? std::string("nullptr") : std::string((char*)values)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_22(mht_22_v, 409, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrBoolList");

  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  forward_op_.attrs.Set(attr_name,
                        gtl::ArraySlice<const bool>(b.get(), num_values));
  return parent_op_->SetAttrBoolList(attr_name, values, num_values);
}
Status TapeOperation::SetAttrShapeList(const char* attr_name,
                                       const int64_t** dims,
                                       const int* num_dims, int num_values) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_23(mht_23_v, 424, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrShapeList");

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
  forward_op_.attrs.Set(
      attr_name, gtl::ArraySlice<TensorShapeProto>(proto.get(), num_values));
  return parent_op_->SetAttrShapeList(attr_name, dims, num_dims, num_values);
}
Status TapeOperation::SetAttrFunctionList(
    const char* attr_name, absl::Span<const AbstractOperation*> values) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_24(mht_24_v, 454, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::SetAttrFunctionList");

  return tensorflow::errors::Unimplemented(
      "SetAttrFunctionList has not been "
      "implemented yet.");
}
AbstractOperation* TapeOperation::GetBackingOperation() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_25(mht_25_v, 462, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::GetBackingOperation");
 return parent_op_; }
Status TapeOperation::Execute(absl::Span<AbstractTensorHandle*> retvals,
                              int* num_retvals) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSgradientsPStapePStape_operationDTcc mht_26(mht_26_v, 467, "", "./tensorflow/c/experimental/gradients/tape/tape_operation.cc", "TapeOperation::Execute");

  TF_RETURN_IF_ERROR(parent_op_->Execute(retvals, num_retvals));
  for (int i = 0; i < *num_retvals; i++) {
    // TODO(srbs): Manage refcount of ForwardOperation's inputs/outputs.
    forward_op_.outputs.push_back(retvals[i]);
  }
  // TODO(b/166669239): This is needed to support AttrBuilder::Get for string
  // attributes. Number type attrs and DataType attrs work fine without this.
  // Consider getting rid of this and making the behavior between number types
  // and string consistent.
  forward_op_.attrs.BuildNodeDef();
  // TODO(b/170307493): Populate skip_input_indices here.
  std::unique_ptr<GradientFunction> backward_fn;
  TF_RETURN_IF_ERROR(registry_.Lookup(forward_op_, &backward_fn));
  tape_->RecordOperation(forward_op_.inputs, forward_op_.outputs,
                         backward_fn.release(), parent_op_->Name());
  return Status::OK();
}

}  // namespace gradients
}  // namespace tensorflow
