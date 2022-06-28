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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

bool HloInputOutputAliasConfig::OutputHasAlias(
    const ShapeIndex& output_index) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::OutputHasAlias");

  return alias_.element(output_index).has_value();
}

Status HloInputOutputAliasConfig::SetUpAlias(
    const ShapeIndex& output_index, int64_t param_number,
    const ShapeIndex& param_index,
    HloInputOutputAliasConfig::AliasKind must_alias) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_1(mht_1_v, 203, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::SetUpAlias");

  TF_RET_CHECK(ShapeUtil::IndexIsValid(alias_.shape(), output_index))
      << "Trying to set up alias at " << output_index.ToString()
      << " which is an invalid index for shape "
      << ShapeUtil::HumanString(alias_.shape());
  TF_RET_CHECK(param_number >= 0) << param_number;
  TF_RET_CHECK(!OutputHasAlias(output_index))
      << "Output index " << output_index << " already has an alias setup";
  // Output can't be aliased with multiple parameters.
  TF_RET_CHECK(!alias_.element(output_index)) << absl::StrFormat(
      "Trying to set up output alias for param %lld at %s but failed: output "
      "index %s is already aliased with param %lld at %s",
      param_number, param_index.ToString(), output_index.ToString(),
      alias_.element(output_index)->parameter_number,
      alias_.element(output_index)->parameter_index.ToString());
  (*alias_.mutable_element(output_index)) =
      Alias(param_number, param_index, must_alias);
  VLOG(4) << "Set up alias between output index " << output_index.ToString()
          << " and parameter " << param_index << " at index "
          << param_index.ToString();
  return Status::OK();
}

HloInputOutputAliasProto HloInputOutputAliasConfig::ToProto() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_2(mht_2_v, 229, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::ToProto");

  HloInputOutputAliasProto result;
  alias_.ForEachElement(
      [&](const ShapeIndex& index, const absl::optional<Alias>& data) {
        if (data) {
          HloInputOutputAliasProto::AliasEntryProto entry;
          for (int64_t i : index) {
            entry.add_output_shape_index(i);
          }
          entry.set_parameter_number(data->parameter_number);
          for (int64_t i : data->parameter_index) {
            entry.add_parameter_shape_index(i);
          }
          if (data->must_alias()) {
            entry.set_kind(Kind::MUST_ALIAS);
          } else {
            entry.set_kind(Kind::MAY_ALIAS);
          }
          result.add_entries()->Swap(&entry);
        }
      });
  return result;
}

StatusOr<HloInputOutputAliasConfig> HloInputOutputAliasConfig::CreateFromProto(
    Shape output_shape, const HloInputOutputAliasProto& proto) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_3(mht_3_v, 257, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::CreateFromProto");

  HloInputOutputAliasConfig result(std::move(output_shape));
  for (const HloInputOutputAliasProto::AliasEntryProto& entry :
       proto.entries()) {
    ShapeIndex output_index(entry.output_shape_index().begin(),
                            entry.output_shape_index().end());
    int64_t param_number = entry.parameter_number();
    ShapeIndex param_index(entry.parameter_shape_index().begin(),
                           entry.parameter_shape_index().end());
    AliasKind kind = entry.kind() == Kind::MAY_ALIAS ? kMayAlias : kMustAlias;
    TF_RETURN_IF_ERROR(
        result.SetUpAlias(output_index, param_number, param_index, kind));
  }
  return result;
}

const Shape& HloInputOutputAliasConfig::shape() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_4(mht_4_v, 276, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::shape");
 return alias_.shape(); }

std::string HloInputOutputAliasConfig::ToString() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_5(mht_5_v, 281, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::ToString");

  std::vector<std::string> pieces;
  pieces.push_back("HloInputOutputAliasConfig");
  pieces.push_back(
      absl::StrFormat("  Output shape: %s", alias_.shape().ToString()));

  ForEachAlias([&](const ShapeIndex& output_index, const Alias& alias) {
    pieces.push_back(absl::StrFormat(
        "  OutputIndex %s is %saliased with parameter %lld at %s:",
        output_index.ToString(), alias.kind == kMustAlias ? "must-" : "may-",
        alias.parameter_number, alias.parameter_index.ToString()));
  });
  return absl::StrJoin(pieces, "\n");
}

std::string HloInputOutputAliasConfig::ToShortString() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_6(mht_6_v, 299, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::ToShortString");

  std::vector<std::string> pieces;
  for (const auto& p : alias_) {
    const ShapeIndex& index = p.first;
    if (absl::optional<Alias> alias = p.second) {
      pieces.push_back(
          absl::StrFormat("%s: %s", index.ToString(), alias->ToString()));
    }
  }
  return absl::StrJoin(pieces, ", ");
}

bool HloInputOutputAliasConfig::ParameterMustAlias(
    int64_t param_number, const ShapeIndex& param_index) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_7(mht_7_v, 315, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::ParameterMustAlias");

  bool result = false;
  alias_.ForEachElement(
      [&](const xla::ShapeIndex&, absl::optional<Alias> alias) {
        if (alias && alias->parameter_number == param_number &&
            alias->parameter_index == param_index && alias->must_alias()) {
          result = true;
        }
      });
  return result;
}

absl::optional<ShapeIndex> HloInputOutputAliasConfig::GetAliasedOutput(
    int64_t param_number, const ShapeIndex& param_index) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_8(mht_8_v, 331, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::GetAliasedOutput");

  absl::optional<ShapeIndex> output;
  alias_.ForEachElement(
      [&](const xla::ShapeIndex& output_index, absl::optional<Alias> alias) {
        if (alias && alias->parameter_number == param_number &&
            alias->parameter_index == param_index) {
          output = output_index;
        }
      });
  return output;
}

absl::optional<HloInputOutputAliasConfig::Alias>
HloInputOutputAliasConfig::GetAliasedParameter(
    const ShapeIndex& output_index) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_9(mht_9_v, 348, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::GetAliasedParameter");

  CHECK(ShapeUtil::IndexIsValid(alias_.shape(), output_index))
      << ToString() << " " << alias_.shape().ToString() << " " << output_index;
  return alias_.element(output_index);
}

void HloInputOutputAliasConfig::ForEachAlias(AliasFn fn) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_10(mht_10_v, 357, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::ForEachAlias");

  alias_.ForEachElement(
      [&](const ShapeIndex& output_index, absl::optional<Alias> aliased) {
        if (aliased) {
          fn(output_index, *aliased);
        }
      });
}

Status HloInputOutputAliasConfig::ForEachAliasWithStatus(
    AliasFnWithStatus fn) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_11(mht_11_v, 370, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::ForEachAliasWithStatus");

  return alias_.ForEachElementWithStatus(
      [&](const ShapeIndex& output_index, absl::optional<Alias> aliased) {
        if (aliased) {
          TF_RETURN_IF_ERROR(fn(output_index, *aliased));
        }
        return Status::OK();
      });
}

Status HloInputOutputAliasConfig::Verify(
    const HloModule& module,
    std::function<int64_t(const Shape&)> size_func) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_12(mht_12_v, 385, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "HloInputOutputAliasConfig::Verify");

  std::vector<ShapeTree<bool>> param_has_seen;
  const HloComputation* entry = module.entry_computation();
  for (int64_t i = 0; i < entry->num_parameters(); ++i) {
    HloInstruction* param = entry->parameter_instruction(i);
    param_has_seen.emplace_back(param->shape());
  }
  return ForEachAliasWithStatus([&](const ShapeIndex& output_index,
                                    const Alias& alias) -> Status {
    const HloInstruction* root = entry->root_instruction();

    TF_RET_CHECK(0 <= alias.parameter_number);
    TF_RET_CHECK(entry->num_parameters() > alias.parameter_number);
    const Shape& param_shape =
        entry->parameter_instruction(alias.parameter_number)->shape();
    const Shape& output_shape = root->shape();
    TF_RET_CHECK(ShapeUtil::IndexIsValid(param_shape, alias.parameter_index));
    TF_RET_CHECK(ShapeUtil::IndexIsValid(output_shape, output_index));

    const Shape& param_subshape =
        ShapeUtil::GetSubshape(param_shape, alias.parameter_index);
    const Shape& output_subshape =
        ShapeUtil::GetSubshape(output_shape, output_index);
    TF_RET_CHECK(LayoutUtil::IsDenseArray(param_subshape));
    TF_RET_CHECK(LayoutUtil::IsDenseArray(output_subshape));

    if (size_func(param_subshape) != size_func(output_subshape)) {
      return InternalError(
          "Expected aliased input %lld at index %s and output at index %s to "
          "have the same size. Input sub-shape is %s with size %lld, output "
          "sub-shape is %s with size %lld",
          alias.parameter_number, alias.parameter_index.ToString(),
          output_index.ToString(),
          ShapeUtil::HumanStringWithLayout(param_subshape),
          size_func(param_subshape),
          ShapeUtil::HumanStringWithLayout(output_subshape),
          size_func(output_subshape));
    }

    // Check each alias.parameter_number and alias.parameter_index pair only
    // show up once. No input can be aliased with output buffers.
    TF_RET_CHECK(param_has_seen[alias.parameter_number].element(
                     alias.parameter_index) == false);
    *(param_has_seen[alias.parameter_number].mutable_element(
        alias.parameter_index)) = true;
    return Status::OK();
  });
}

std::ostream& operator<<(std::ostream& out,
                         const HloInputOutputAliasConfig& config) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_input_output_alias_configDTcc mht_13(mht_13_v, 438, "", "./tensorflow/compiler/xla/service/hlo_input_output_alias_config.cc", "operator<<");

  out << config.ToString();
  return out;
}
}  // namespace xla
