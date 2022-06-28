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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc() {
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

#include "tensorflow/compiler/xla/service/dynamic_window_utils.h"

#include <string>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {
namespace {
// HloOp wraps an instuction pointer to do arithmetic based on operator
// overloading.
//
// TODO(yunxing): This is only used internally to this file to provide a
// convenient way to do operator overloadding.  Find out an idiom and merge this
// with hlo_creation_utils.
class HloOp {
 public:
  HloOp() = default;
  explicit HloOp(HloInstruction* inst) : inst_(inst) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "HloOp");
}
  void SetName(const std::string& name) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "SetName");

    inst_->SetAndSanitizeName(name);
    if (inst_->GetModule() != nullptr) {
      inst_->UniquifyName(&inst_->GetModule()->instruction_name_uniquer());
    }
  }
  HloInstruction* get() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_2(mht_2_v, 221, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "get");
 return inst_; }

 private:
  HloInstruction* inst_ = nullptr;
};
HloOp BinaryOp(HloOp x, HloOp y, HloOpcode opcode,
               const std::string& name = "") {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_3(mht_3_v, 230, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "BinaryOp");

  CHECK_EQ(x.get()->parent(), y.get()->parent());
  Shape binary_op_shape =
      ShapeInference::InferBinaryOpShape(opcode, x.get(), y.get()).ValueOrDie();
  return HloOp(x.get()->parent()->AddInstruction(
      HloInstruction::CreateBinary(binary_op_shape, opcode, x.get(), y.get()),
      name));
}
HloOp operator+(HloOp x, HloOp y) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_4(mht_4_v, 241, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "+");
 return BinaryOp(x, y, HloOpcode::kAdd); }

HloOp operator-(HloOp x, HloOp y) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_5(mht_5_v, 246, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "-");

  return BinaryOp(x, y, HloOpcode::kSubtract);
}

HloOp operator*(HloOp x, HloOp y) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_6(mht_6_v, 253, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "*");

  return BinaryOp(x, y, HloOpcode::kMultiply);
}

HloOp operator/(HloOp x, HloOp y) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_7(mht_7_v, 260, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "/");
 return BinaryOp(x, y, HloOpcode::kDivide); }

HloOp Maximum(HloOp x, HloOp y, const std::string& name = "") {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_8(mht_8_v, 265, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "Maximum");

  return BinaryOp(x, y, HloOpcode::kMaximum, name);
}

template <typename NativeT>
HloOp ConstantR0(HloComputation* comp, NativeT value,
                 const std::string& name = "") {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_9(mht_9_v, 274, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "ConstantR0");

  return HloOp(comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<NativeT>(value)),
      name));
}

template <typename NativeT>
HloOp One(HloComputation* comp) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_10(mht_10_v, 284, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "One");

  return ConstantR0<NativeT>(comp, 1, "one");
}

template <typename NativeT>
HloOp Zero(HloComputation* comp) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_11(mht_11_v, 292, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "Zero");

  return ConstantR0<NativeT>(comp, 0, "zero");
}

HloOp EffectiveFilterSize(HloComputation* comp, int64_t window_size,
                          int64_t window_dilation) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_12(mht_12_v, 300, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "EffectiveFilterSize");

  return ConstantR0<int32_t>(comp, (window_size - 1) * window_dilation + 1,
                             "effective_filter_size");
}
}  // namespace

DynamicWindowDims GetWindowedOutputSize(HloInstruction* input_size,
                                        int64_t window_size,
                                        int64_t window_dilation,
                                        int64_t window_stride,
                                        PaddingType padding_type) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_13(mht_13_v, 313, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "GetWindowedOutputSize");

  HloComputation* comp = input_size->parent();
  DynamicWindowDims result;

  HloOp stride = ConstantR0<int32_t>(comp, window_stride, "stride");
  HloOp effective_filter_size =
      EffectiveFilterSize(comp, window_size, window_dilation);
  if (padding_type == PaddingType::PADDING_VALID) {
    HloOp output =
        (HloOp(input_size) + stride - effective_filter_size) / stride;
    result.output_size = output.get();
    result.padding_before = Zero<int32_t>(comp).get();
  } else if (padding_type == PaddingType::PADDING_SAME) {
    HloOp output = (HloOp(input_size) + stride - One<int32_t>(comp)) / stride;
    HloOp padding_needed = Maximum(
        Zero<int32_t>(comp), (output - One<int32_t>(comp)) * stride +
                                 effective_filter_size - HloOp(input_size));
    HloOp padding_before = padding_needed / ConstantR0<int32_t>(comp, 2);
    result.padding_before = padding_before.get();
    result.output_size = output.get();
  }

  return result;
}

DynamicWindowDims GetWindowedInputGradSize(HloInstruction* input_size,
                                           int64_t window_size,
                                           int64_t window_dilation,
                                           int64_t window_stride,
                                           PaddingType padding_type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_window_utilsDTcc mht_14(mht_14_v, 345, "", "./tensorflow/compiler/xla/service/dynamic_window_utils.cc", "GetWindowedInputGradSize");

  HloComputation* comp = input_size->parent();
  DynamicWindowDims result;
  HloOp effective_filter_size =
      ConstantR0<int32_t>(comp, (window_size - 1) * window_dilation + 1);
  HloOp stride = ConstantR0<int32_t>(comp, window_stride);
  DynamicWindowDims forward_dims = GetWindowedOutputSize(
      input_size, window_size, window_dilation, window_stride, padding_type);
  HloOp output_size =
      (HloOp(forward_dims.output_size) - One<int32_t>(comp)) * stride +
      One<int32_t>(comp);
  HloOp padding_before = effective_filter_size - One<int32_t>(comp) -
                         HloOp(forward_dims.padding_before);
  result.output_size = output_size.get();
  result.padding_before = padding_before.get();
  return result;
}
}  // namespace xla
