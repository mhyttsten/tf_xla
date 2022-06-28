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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.h"

#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/kernel/attr_util.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_frame.h"  // from @tf_runtime

namespace tensorflow {

//////////////////////////////////////////////////////////////////////
// OpKernel interface.
//////////////////////////////////////////////////////////////////////
TFRTOpKernelConstruction::TFRTOpKernelConstruction(
    const tfrt::OpAttrsRef& attributes)
    : attributes_(std::move(attributes)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::TFRTOpKernelConstruction");
}

Status MissingAttributeError(StringPiece attr_name) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_1(mht_1_v, 210, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "MissingAttributeError");

  return errors::InvalidArgument("Missing attribute: ", attr_name);
}

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         std::string* value) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_2(mht_2_v, 219, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::GetAttr");

  tfrt::string_view view;
  bool success = attributes_.GetString(
      llvm::StringRef(attr_name.data(), attr_name.size()), &view);
  if (!success) {
    return MissingAttributeError(attr_name);
  }
  *value = view.str();
  return Status::OK();
}

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         DataType* value) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_3(mht_3_v, 235, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::GetAttr");

  tfrt::OpAttrType attrtype;
  bool success = attributes_.Get<tfrt::OpAttrType>(
      llvm::StringRef(attr_name.data(), attr_name.size()), &attrtype);
  if (!success) {
    return MissingAttributeError(attr_name);
  }
  *value = tfd::ConvertToTfDataType(attrtype);
  return Status::OK();
}

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         Padding* value) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_4(mht_4_v, 251, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::GetAttr");

  std::string padding_str;
  TF_RETURN_IF_ERROR(GetAttr<std::string>(attr_name, &padding_str));
  return GetPaddingFromString(padding_str, value);
}

template <>
Status TFRTOpKernelConstruction::GetAttr(StringPiece attr_name,
                                         std::vector<int32>* value) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_5(mht_5_v, 262, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::GetAttr");

  llvm::ArrayRef<int32> arrayref;
  bool success = attributes_.GetArray<int32>(
      llvm::StringRef(attr_name.data(), attr_name.size()), &arrayref);
  if (!success) {
    return MissingAttributeError(attr_name);
  }
  *value = arrayref;
  return Status::OK();
}

void TFRTOpKernelConstruction::CtxFailure(const Status& s) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_6(mht_6_v, 276, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::CtxFailure");

  error_ = tfrt::MakeStatusString(s);
}

void TFRTOpKernelConstruction::CtxFailureWithWarning(const Status& s) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_7(mht_7_v, 283, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::CtxFailureWithWarning");

  CtxFailure(s);
}

namespace {
std::string FillFailureMessage(const char* file, int line, const Status& s) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_8(mht_8_v, 292, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "FillFailureMessage");

  std::string error;
  llvm::raw_string_ostream sstr(error);
  sstr << "OP_REQUIRES failed at " << file << ":" << line << " : "
       << tfrt::MakeStatusString(s);
  sstr.str();
  return error;
}
}  // namespace

void TFRTOpKernelConstruction::CtxFailure(const char* file, int line,
                                          const Status& s) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_9(mht_9_v, 307, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::CtxFailure");

  error_ = FillFailureMessage(file, line, s);
}

void TFRTOpKernelConstruction::CtxFailureWithWarning(const char* file, int line,
                                                     const Status& s) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_10(mht_10_v, 316, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::CtxFailureWithWarning");

  CtxFailure(file, line, s);
}

const llvm::Optional<std::string>& TFRTOpKernelConstruction::error() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_11(mht_11_v, 323, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelConstruction::error");

  return error_;
}

TFRTOpKernelContext::TFRTOpKernelContext(
    llvm::ArrayRef<tfrt::RCReference<tfrt::AsyncValue>> inputs, int num_outputs,
    const TFRTOpMeta* op_meta, tfrt::HostContext* host)
    : inputs_(inputs),
      op_meta_(op_meta),
      outputs_(num_outputs),
      eigen_host_context_(host) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_12(mht_12_v, 336, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::TFRTOpKernelContext");
}

const Tensor& TFRTOpKernelContext::output(int index) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_13(mht_13_v, 341, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::output");
 return outputs_[index]; }

const llvm::Optional<std::string>& TFRTOpKernelContext::error() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_14(mht_14_v, 346, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::error");

  return error_;
}

bool TFRTOpKernelContext::ValidateInputsAreSameShape(TFRTOpKernel* op) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_15(mht_15_v, 353, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::ValidateInputsAreSameShape");

  // TODO(lauj) Check shapes.
  return true;
}

const Tensor& TFRTOpKernelContext::input(int index) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_16(mht_16_v, 361, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::input");

  return inputs_[index]->get<Tensor>();
}

int TFRTOpKernelContext::num_inputs() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_17(mht_17_v, 368, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::num_inputs");
 return inputs_.size(); }

int TFRTOpKernelContext::num_outputs() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_18(mht_18_v, 373, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::num_outputs");
 return outputs_.size(); }

void TFRTOpKernelContext::set_output(int index, const Tensor& tensor) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_19(mht_19_v, 378, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::set_output");

  outputs_[index] = tensor;
}

Status TFRTOpKernelContext::allocate_temp(DataType type,
                                          const TensorShape& shape,
                                          Tensor* out_temp) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_20(mht_20_v, 387, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::allocate_temp");

  *out_temp = Tensor(type, shape);
  return Status::OK();
}

Status TFRTOpKernelContext::allocate_output(int index, const TensorShape& shape,
                                            Tensor** tensor) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_21(mht_21_v, 396, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::allocate_output");

  // Fetch output DataType from the op's TFRTOpMeta.
  DataType output_type = op_meta_->output_type(index);
  outputs_[index] = Tensor(output_type, shape);
  *tensor = &outputs_[index];
  return Status::OK();
}

DataType TFRTOpKernelContext::expected_output_dtype(int i) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_22(mht_22_v, 407, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::expected_output_dtype");

  return op_meta_->output_type(i);
}

void TFRTOpKernelContext::CtxFailure(const Status& s) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_23(mht_23_v, 414, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::CtxFailure");

  error_ = s.error_message();
}
void TFRTOpKernelContext::CtxFailureWithWarning(const Status& s) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_24(mht_24_v, 420, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::CtxFailureWithWarning");

  CtxFailure(s);
}
void TFRTOpKernelContext::CtxFailure(const char* file, int line,
                                     const Status& s) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_25(mht_25_v, 428, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::CtxFailure");

  error_ = FillFailureMessage(file, line, s);
}
void TFRTOpKernelContext::CtxFailureWithWarning(const char* file, int line,
                                                const Status& s) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("file: \"" + (file == nullptr ? std::string("nullptr") : std::string((char*)file)) + "\"");
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_26(mht_26_v, 436, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::CtxFailureWithWarning");

  CtxFailure(file, line, s);
}

template <>
const Eigen::ThreadPoolDevice& TFRTOpKernelContext::eigen_device() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_27(mht_27_v, 444, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelContext::eigen_device");

  return eigen_host_context_.Device();
}

//////////////////////////////////////////////////////////////////////
// Forwarding op metadata.
//////////////////////////////////////////////////////////////////////
TFRTOpMeta::TFRTOpMeta(std::vector<DataType> output_types)
    : output_types_(std::move(output_types)) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_28(mht_28_v, 455, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMeta::TFRTOpMeta");
}

DataType TFRTOpMeta::output_type(int index) const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_29(mht_29_v, 460, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMeta::output_type");

  return output_types_[index];
}

TFRTOpMetaBuilder::TFRTOpMetaBuilder(StringPiece op_name) : op_name_(op_name) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_30(mht_30_v, 467, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMetaBuilder::TFRTOpMetaBuilder");
}

namespace {

DataType ParseInputOutputSpec(StringPiece spec) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_31(mht_31_v, 474, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "ParseInputOutputSpec");

  std::vector<absl::string_view> name_type =
      absl::StrSplit(spec, absl::MaxSplits(':', 2));
  DataType data_type;
  bool success =
      DataTypeFromString(absl::StripAsciiWhitespace(name_type[1]), &data_type);
  assert(success && "Failed to parse DataType");
  (void)success;
  return data_type;
}

}  // anonymous namespace

TFRTOpMetaBuilder& TFRTOpMetaBuilder::Output(StringPiece output_spec) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_32(mht_32_v, 490, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMetaBuilder::Output");

  output_types_.push_back(ParseInputOutputSpec(output_spec));
  return *this;
}

TFRTOpMetaBuilder& TFRTOpMetaBuilder::Input(StringPiece input_spec) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_33(mht_33_v, 498, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMetaBuilder::Input");

  return *this;
}

TFRTOpMetaBuilder& TFRTOpMetaBuilder::Attr(StringPiece attr_spec) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_34(mht_34_v, 505, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMetaBuilder::Attr");

  return *this;
}

const string& TFRTOpMetaBuilder::op_name() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_35(mht_35_v, 512, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMetaBuilder::op_name");
 return op_name_; }

TFRTOpMeta TFRTOpMetaBuilder::BuildMeta() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_36(mht_36_v, 517, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMetaBuilder::BuildMeta");

  return TFRTOpMeta(output_types_);
}

TFRTOpMetaMap::TFRTOpMetaMap() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_37(mht_37_v, 524, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMetaMap::TFRTOpMetaMap");
}

void TFRTOpMetaMap::RegisterOpMeta(const TFRTOpMetaBuilder& op_builder) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_38(mht_38_v, 529, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMetaMap::RegisterOpMeta");

  auto insert_result = op_metas_.insert(
      std::make_pair(op_builder.op_name(), op_builder.BuildMeta()));
  assert(insert_result.second && "Multiple registrations for the same op_name");
  (void)insert_result;
}

const TFRTOpMeta* TFRTOpMetaMap::GetOpMeta(StringPiece op_name) const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_39(mht_39_v, 539, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpMetaMap::GetOpMeta");

  auto it = op_metas_.find(llvm::StringRef(op_name.data(), op_name.size()));
  if (it == op_metas_.end()) return nullptr;

  return &it->second;
}

TFRTOpRegisterer::TFRTOpRegisterer(const TFRTOpMetaBuilder& op_builder) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_40(mht_40_v, 549, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpRegisterer::TFRTOpRegisterer");

  tfrt_forwarding_op_meta_map->RegisterOpMeta(op_builder);
}

llvm::ManagedStatic<TFRTOpMetaMap> tfrt_forwarding_op_meta_map;

llvm::ManagedStatic<TFRTOpKernelFactories> tfrt_forwarding_kernel_factories;

//////////////////////////////////////////////////////////////////////
// Forwarding kernel registration.
//////////////////////////////////////////////////////////////////////

TFRTOpKernelFactories::TFRTOpKernelFactories() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_41(mht_41_v, 564, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelFactories::TFRTOpKernelFactories");
}

void TFRTOpKernelFactories::RegisterFactory(StringPiece kernel_class_name,
                                            TFRTOpKernelReg kernel_info) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_42(mht_42_v, 570, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelFactories::RegisterFactory");

  factories_[std::string(kernel_class_name)].push_back(kernel_info);
}

// Returns true if kernel attributes match given type constraints.
Status ValidKernelAttr(StringPiece kernel_class_name,
                       TFRTOpKernelConstruction* construction,
                       const llvm::StringMap<DataType>& constraints) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_43(mht_43_v, 580, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "ValidKernelAttr");

  for (const auto& constraint : constraints) {
    auto attr_name = std::string(constraint.first());
    DataType type;
    Status s = construction->GetAttr(attr_name, &type);
    if (!s.ok()) {
      return errors::InvalidArgument(
          "Kernel ", kernel_class_name,
          " has constraint for unset tfdtype attribute ", attr_name, ".");
    }
    if (type != constraint.second) {
      return errors::InvalidArgument(
          "Kernel ", kernel_class_name, " with type constraint ", attr_name,
          ": ", DataTypeString(constraint.second),
          " does not match attribute type ", DataTypeString(type), ".");
    }
  }
  return Status::OK();
}

std::unique_ptr<TFRTOpKernel> TFRTOpKernelFactories::CreateKernel(
    StringPiece kernel_class_name,
    TFRTOpKernelConstruction* op_kernel_construction) const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSkernelPStfrt_op_kernelDTcc mht_44(mht_44_v, 605, "", "./tensorflow/core/runtime_fallback/kernel/tfrt_op_kernel.cc", "TFRTOpKernelFactories::CreateKernel");

  auto it = factories_.find(std::string(kernel_class_name));
  if (it == factories_.end()) {
    // Could not find kernel in the registry
    op_kernel_construction->CtxFailure(errors::NotFound(
        "Could not find kernel ", kernel_class_name, " in the registry."));
    return std::unique_ptr<TFRTOpKernel>(nullptr);
  }
  Status status;
  for (const auto& kernel_info : it->second) {
    Status s = ValidKernelAttr(kernel_class_name, op_kernel_construction,
                               kernel_info.type_constraints);
    if (s.ok()) {
      return kernel_info.callback(op_kernel_construction);
    }
    status.Update(s);
  }
  // No valid kernel found
  op_kernel_construction->CtxFailure(status);
  return std::unique_ptr<TFRTOpKernel>(nullptr);
}

}  // namespace tensorflow
