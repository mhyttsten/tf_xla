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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc() {
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

#include "tensorflow/compiler/xla/service/dynamic_parameter_binding.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

Status DynamicParameterBinding::Bind(
    const DynamicParameter& dynamic_parameter,
    const DynamicDimension& dynamic_dimension) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/xla/service/dynamic_parameter_binding.cc", "DynamicParameterBinding::Bind");

  auto result = bindings_.emplace(dynamic_dimension, dynamic_parameter);
  TF_RET_CHECK(result.second);
  return Status::OK();
}

absl::optional<DynamicParameterBinding::DynamicParameter>
DynamicParameterBinding::GetBinding(
    const DynamicDimension& dynamic_dimension) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc mht_1(mht_1_v, 206, "", "./tensorflow/compiler/xla/service/dynamic_parameter_binding.cc", "DynamicParameterBinding::GetBinding");

  auto param_iter = bindings_.find(dynamic_dimension);
  if (param_iter == bindings_.end()) {
    return absl::nullopt;
  }
  return param_iter->second;
}

DynamicParameterBindingProto DynamicParameterBinding::ToProto() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc mht_2(mht_2_v, 217, "", "./tensorflow/compiler/xla/service/dynamic_parameter_binding.cc", "DynamicParameterBinding::ToProto");

  DynamicParameterBindingProto result;
  for (const auto& binding : bindings_) {
    const DynamicDimension& dynamic_dimension = binding.first;
    const DynamicParameter& dynamic_param = binding.second;
    DynamicParameterBindingProto::Binding binding_proto;
    binding_proto.set_dynamic_param_num(dynamic_param.parameter_num);
    for (int64_t i : dynamic_param.parameter_index) {
      binding_proto.add_dynamic_param_index(i);
    }

    binding_proto.set_target_param_num(dynamic_dimension.parameter_num);

    for (int64_t i : dynamic_dimension.parameter_index) {
      binding_proto.add_target_param_index(i);
    }

    binding_proto.set_target_param_dim_num(dynamic_dimension.dimension);
    result.add_entries()->Swap(&binding_proto);
  }
  return result;
}

StatusOr<DynamicParameterBinding> DynamicParameterBinding::CreateFromProto(
    const DynamicParameterBindingProto& proto) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc mht_3(mht_3_v, 244, "", "./tensorflow/compiler/xla/service/dynamic_parameter_binding.cc", "DynamicParameterBinding::CreateFromProto");

  DynamicParameterBinding result;
  for (const DynamicParameterBindingProto::Binding& binding : proto.entries()) {
    int64_t dynamic_param_num = binding.dynamic_param_num();
    ShapeIndex dynamic_param_index(binding.dynamic_param_index().begin(),
                                   binding.dynamic_param_index().end());
    int64_t target_param_num = binding.target_param_num();
    ShapeIndex target_param_index(binding.target_param_index().begin(),
                                  binding.target_param_index().end());
    int64_t target_dim_num = binding.target_param_dim_num();

    TF_RETURN_IF_ERROR(
        result.Bind(DynamicParameter{dynamic_param_num, dynamic_param_index},
                    DynamicDimension{target_param_num, target_param_index,
                                     target_dim_num}));
  }

  return result;
}

std::string DynamicParameterBinding::ToString() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc mht_4(mht_4_v, 267, "", "./tensorflow/compiler/xla/service/dynamic_parameter_binding.cc", "DynamicParameterBinding::ToString");

  std::vector<std::string> pieces;
  pieces.push_back("DynamicParameterBinding: ");
  for (const auto& binding : bindings_) {
    const DynamicDimension& dynamic_dimension = binding.first;
    const DynamicParameter& dynamic_param = binding.second;
    pieces.push_back(absl::StrFormat(
        " -- Input param number %lld at %s has dim %lld as dynamic"
        " dimension, which is represented by param number %lld at "
        "%s",
        dynamic_dimension.parameter_num,
        dynamic_dimension.parameter_index.ToString(),
        dynamic_dimension.dimension, dynamic_param.parameter_num,
        dynamic_param.parameter_index.ToString()));
  }
  return absl::StrJoin(pieces, "\n");
}

Status DynamicParameterBinding::ForEachBinding(BindingFn fn) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc mht_5(mht_5_v, 288, "", "./tensorflow/compiler/xla/service/dynamic_parameter_binding.cc", "DynamicParameterBinding::ForEachBinding");

  for (const auto& binding : bindings_) {
    TF_RETURN_IF_ERROR(fn(binding.second, binding.first));
  }
  return Status::OK();
}

Status DynamicParameterBinding::Verify(const HloModule& module) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc mht_6(mht_6_v, 298, "", "./tensorflow/compiler/xla/service/dynamic_parameter_binding.cc", "DynamicParameterBinding::Verify");

  const HloComputation* entry = module.entry_computation();
  return ForEachBinding([&](const DynamicParameter& dynamic_parameter,
                            const DynamicDimension& dynamic_dimension)
                            -> Status {
    TF_RET_CHECK(dynamic_parameter.parameter_num >= 0 &&
                 dynamic_parameter.parameter_num < entry->num_parameters());
    TF_RET_CHECK(dynamic_dimension.parameter_num < entry->num_parameters());
    TF_RET_CHECK(ShapeUtil::IndexIsValid(
        entry->parameter_instruction(dynamic_parameter.parameter_num)->shape(),
        dynamic_parameter.parameter_index));
    TF_RET_CHECK(ShapeUtil::IndexIsValid(
        entry->parameter_instruction(dynamic_dimension.parameter_num)->shape(),
        dynamic_dimension.parameter_index));
    TF_RET_CHECK(
        dynamic_dimension.dimension <
        ShapeUtil::GetSubshape(
            entry->parameter_instruction(dynamic_dimension.parameter_num)
                ->shape(),
            dynamic_dimension.parameter_index)
            .rank());
    return Status::OK();
  });
}

std::ostream& operator<<(std::ostream& out,
                         const DynamicParameterBinding& binding) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTcc mht_7(mht_7_v, 327, "", "./tensorflow/compiler/xla/service/dynamic_parameter_binding.cc", "operator<<");

  out << binding.ToString();
  return out;
}

}  // namespace xla
