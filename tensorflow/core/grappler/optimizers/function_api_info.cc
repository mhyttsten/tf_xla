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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc() {
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

#include "tensorflow/core/grappler/optimizers/function_api_info.h"

#include <string>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
FunctionApiInfo::FunctionApiInfo() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionApiInfo::FunctionApiInfo");
}
FunctionApiInfo::~FunctionApiInfo() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_1(mht_1_v, 199, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionApiInfo::~FunctionApiInfo");
}

Status FunctionApiInfo::Init(const FunctionDef& function_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_2(mht_2_v, 204, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionApiInfo::Init");

  function_type_ = FunctionApiInfo::FunctionType::INFERENCE;
  for (const auto& attr : function_def.attr()) {
    if (attr.first == "api_preferred_device") {
      preferred_device_ = attr.second.s();
    }
    if (attr.first == "api_implements") {
      interface_name_ = attr.second.s();
    }
    if (attr.first == "forward_function_name") {
      function_type_ = FunctionApiInfo::FunctionType::BACKWARD;
      pairing_function_name_ = attr.second.s();
    }
    if (attr.first == "backward_function_name") {
      function_type_ = FunctionApiInfo::FunctionType::FORWARD;
      pairing_function_name_ = attr.second.s();
    }
  }

  input_arg_dtypes_.reserve(function_def.signature().input_arg_size());
  for (const auto& input_arg : function_def.signature().input_arg()) {
    input_arg_dtypes_.emplace_back(input_arg.type());
  }
  output_arg_dtypes_.reserve(function_def.signature().output_arg_size());
  for (const auto& output_arg : function_def.signature().output_arg()) {
    output_arg_dtypes_.emplace_back(output_arg.type());
  }

  if (interface_name_.empty() && !preferred_device_.empty()) {
    return errors::InvalidArgument(
        "Function '", function_def.signature().name(),
        "' has a preferred device, but does not implement an interface");
  }
  return Status::OK();
}

const string& FunctionApiInfo::preferred_device() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_3(mht_3_v, 243, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionApiInfo::preferred_device");

  return preferred_device_;
}

const string& FunctionApiInfo::interface_name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_4(mht_4_v, 250, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionApiInfo::interface_name");

  return interface_name_;
}

const FunctionApiInfo::FunctionType FunctionApiInfo::function_type() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_5(mht_5_v, 257, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionApiInfo::function_type");

  return function_type_;
}

const string& FunctionApiInfo::pairing_function_name() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_6(mht_6_v, 264, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionApiInfo::pairing_function_name");

  return pairing_function_name_;
}

const DataTypeVector& FunctionApiInfo::input_arg_dtypes() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_7(mht_7_v, 271, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionApiInfo::input_arg_dtypes");

  return input_arg_dtypes_;
}

const DataTypeVector& FunctionApiInfo::output_arg_dtypes() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_8(mht_8_v, 278, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionApiInfo::output_arg_dtypes");

  return output_arg_dtypes_;
}

FunctionLibraryApiInfo::FunctionLibraryApiInfo() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_9(mht_9_v, 285, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionLibraryApiInfo::FunctionLibraryApiInfo");
}
FunctionLibraryApiInfo::~FunctionLibraryApiInfo() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_10(mht_10_v, 289, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionLibraryApiInfo::~FunctionLibraryApiInfo");
}

namespace {
bool IsSameArgDef(const OpDef::ArgDef& arg1, const OpDef::ArgDef& arg2) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_11(mht_11_v, 295, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "IsSameArgDef");

  if (arg1.type() != arg2.type()) return false;
  if (arg1.type_attr() != arg2.type_attr()) return false;
  if (arg1.number_attr() != arg2.number_attr()) return false;
  if (arg1.type_list_attr() != arg2.type_list_attr()) return false;
  if (arg1.is_ref() != arg2.is_ref()) return false;
  return true;
}

bool IsSameSignature(const FunctionDef& f1, const FunctionDef& f2,
                     const bool check_inputs, const bool check_outputs) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_12(mht_12_v, 308, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "IsSameSignature");

  const auto& sig1 = f1.signature();
  const auto& sig2 = f2.signature();
  // Functions have positional semantics, so we don't check for names.
  if (check_inputs) {
    if (sig1.input_arg_size() != sig2.input_arg_size()) return false;
    for (int k = 0; k < sig1.input_arg_size(); ++k) {
      if (!IsSameArgDef(sig1.input_arg(k), sig2.input_arg(k))) return false;
    }
  }
  if (check_outputs) {
    if (f1.ret().size() != f2.ret().size()) return false;
    if (sig1.output_arg_size() != sig2.output_arg_size()) return false;
    for (int k = 0; k < sig1.output_arg_size(); ++k) {
      if (!IsSameArgDef(sig1.output_arg(k), sig2.output_arg(k))) return false;
    }
  }
  return true;
}

Status ValidateSignature(const string& interface_name,
                         const std::vector<const FunctionDef*>& equiv_funcs,
                         const FunctionApiInfo::FunctionType function_type) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("interface_name: \"" + interface_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_13(mht_13_v, 334, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "ValidateSignature");

  if (equiv_funcs.size() < 2) return Status::OK();
  for (size_t k = 1; k < equiv_funcs.size(); ++k) {
    const bool check_input =
        (function_type == FunctionApiInfo::FunctionType::INFERENCE ||
         function_type == FunctionApiInfo::FunctionType::FORWARD);
    const bool check_output =
        (function_type == FunctionApiInfo::FunctionType::INFERENCE ||
         function_type == FunctionApiInfo::FunctionType::BACKWARD);
    if (!IsSameSignature(*equiv_funcs[0], *equiv_funcs[k], check_input,
                         check_output)) {
      return errors::InvalidArgument(
          "Functions '", equiv_funcs[0]->signature().name(), "' and '",
          equiv_funcs[k]->signature().name(), "' both implement '",
          interface_name, "' but their signatures do not match.");
    }
  }
  return Status::OK();
}

Status ValidateSignatures(
    const std::unordered_map<string, std::vector<const FunctionDef*>>&
        intf_to_func,
    const FunctionApiInfo::FunctionType function_type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_14(mht_14_v, 360, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "ValidateSignatures");

  for (const auto& item : intf_to_func)
    TF_RETURN_IF_ERROR(
        ValidateSignature(item.first, item.second, function_type));
  return Status::OK();
}
}  // namespace

Status FunctionLibraryApiInfo::Init(
    const FunctionDefLibrary& function_library) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_15(mht_15_v, 372, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionLibraryApiInfo::Init");

  std::unordered_map<string, std::vector<const FunctionDef*>> infer_funcs;
  std::unordered_map<string, std::vector<const FunctionDef*>> fwd_funcs;
  std::unordered_map<string, std::vector<const FunctionDef*>> bwd_funcs;
  for (const auto& function : function_library.function()) {
    std::unique_ptr<FunctionApiInfo> func_info(new FunctionApiInfo);
    TF_RETURN_IF_ERROR(func_info->Init(function));
    // Ignore the function if it does not implement any interface.
    if (func_info->interface_name().empty()) continue;

    const string& function_name = function.signature().name();
    const string& interface_name = func_info->interface_name();
    VLOG(3) << "Got " << func_info->function_type()
            << " function: " << function_name
            << " with interface: " << interface_name;
    switch (func_info->function_type()) {
      case FunctionApiInfo::FunctionType::INFERENCE:
        intf_to_inference_funcs_[interface_name].emplace_back(function_name);
        infer_funcs[interface_name].emplace_back(&function);
        break;
      case FunctionApiInfo::FunctionType::FORWARD:
        intf_to_forward_funcs_[interface_name].emplace_back(function_name);
        fwd_funcs[interface_name].emplace_back(&function);
        break;
      case FunctionApiInfo::FunctionType::BACKWARD:
        intf_to_backward_funcs_[interface_name].emplace_back(function_name);
        bwd_funcs[interface_name].emplace_back(&function);
        break;
      default:
        return errors::InvalidArgument("Unrecognized function type: ",
                                       func_info->function_type());
    }
    func_info_[function_name] = std::move(func_info);
  }
  TF_RETURN_IF_ERROR(ValidateSignatures(
      infer_funcs, FunctionApiInfo::FunctionType::INFERENCE));
  TF_RETURN_IF_ERROR(
      ValidateSignatures(fwd_funcs, FunctionApiInfo::FunctionType::FORWARD));
  TF_RETURN_IF_ERROR(
      ValidateSignatures(bwd_funcs, FunctionApiInfo::FunctionType::BACKWARD));
  return Status::OK();
}

Status FunctionLibraryApiInfo::GetEquivalentImplementations(
    const string& function_name, std::vector<string>* other_functions) const {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_16(mht_16_v, 420, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionLibraryApiInfo::GetEquivalentImplementations");

  const auto func_it = func_info_.find(function_name);
  if (func_it == func_info_.end()) return Status::OK();
  const FunctionApiInfo* func_info = func_it->second.get();

  absl::flat_hash_map<string, std::vector<string>>::const_iterator it;
  switch (func_info->function_type()) {
    case FunctionApiInfo::FunctionType::INFERENCE:
      it = intf_to_inference_funcs_.find(func_info->interface_name());
      break;
    case FunctionApiInfo::FunctionType::FORWARD:
      it = intf_to_forward_funcs_.find(func_info->interface_name());
      break;
    case FunctionApiInfo::FunctionType::BACKWARD:
      it = intf_to_backward_funcs_.find(func_info->interface_name());
      break;
    default:
      return errors::InvalidArgument("Unrecognized function type: ",
                                     func_info->function_type());
  }

  for (const auto& func_name : it->second) {
    if (func_name == function_name) continue;
    other_functions->emplace_back(func_name);
  }
  return Status::OK();
}

const FunctionApiInfo* FunctionLibraryApiInfo::GetApiInfo(
    const string& function_name) const {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("function_name: \"" + function_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTcc mht_17(mht_17_v, 453, "", "./tensorflow/core/grappler/optimizers/function_api_info.cc", "FunctionLibraryApiInfo::GetApiInfo");

  const auto it = func_info_.find(function_name);
  if (it == func_info_.end()) return nullptr;
  return it->second.get();
}

}  // end namespace grappler
}  // end namespace tensorflow
