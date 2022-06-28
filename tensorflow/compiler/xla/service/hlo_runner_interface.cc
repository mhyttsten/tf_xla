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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc() {
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

#include "tensorflow/compiler/xla/service/hlo_runner_interface.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

namespace xla {

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::CreateModuleFromString(const absl::string_view hlo_string,
                                           const DebugOptions& debug_options) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("hlo_string: \"" + std::string(hlo_string.data(), hlo_string.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc mht_0(mht_0_v, 194, "", "./tensorflow/compiler/xla/service/hlo_runner_interface.cc", "HloRunnerInterface::CreateModuleFromString");

  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return ParseAndReturnUnverifiedModule(hlo_string, config);
}

namespace {

// Creates an HloModule from the given proto.
StatusOr<std::unique_ptr<HloModule>> HloProtoToModule(
    const HloProto& proto, const DebugOptions& debug_options) {
  TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                      HloModule::CreateModuleConfigFromProto(proto.hlo_module(),
                                                             debug_options));
  TF_ASSIGN_OR_RETURN(auto module,
                      HloModule::CreateFromProto(proto.hlo_module(), config));
  return std::move(module);
}

}  // namespace

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromBinaryProtoFile(
    const std::string& filename, const DebugOptions& debug_options) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc mht_1(mht_1_v, 221, "", "./tensorflow/compiler/xla/service/hlo_runner_interface.cc", "HloRunnerInterface::ReadModuleFromBinaryProtoFile");

  HloProto proto;
  TF_RETURN_IF_ERROR(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                                 filename, &proto));
  return HloProtoToModule(proto, debug_options);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromTextProtoFile(
    const std::string& filename, const DebugOptions& debug_options) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc mht_2(mht_2_v, 234, "", "./tensorflow/compiler/xla/service/hlo_runner_interface.cc", "HloRunnerInterface::ReadModuleFromTextProtoFile");

  HloProto proto;
  TF_RETURN_IF_ERROR(
      tensorflow::ReadTextProto(tensorflow::Env::Default(), filename, &proto));
  return HloProtoToModule(proto, debug_options);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromHloTextFile(
    const std::string& filename, const DebugOptions& debug_options) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc mht_3(mht_3_v, 247, "", "./tensorflow/compiler/xla/service/hlo_runner_interface.cc", "HloRunnerInterface::ReadModuleFromHloTextFile");

  std::string hlo_string;
  TF_RETURN_IF_ERROR(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                                  filename, &hlo_string));
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return ParseAndReturnUnverifiedModule(hlo_string, config);
}

/*static*/ StatusOr<std::unique_ptr<HloModule>>
HloRunnerInterface::ReadModuleFromModuleBinaryProtofile(
    const std::string& filename, const DebugOptions& debug_options) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("filename: \"" + filename + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc mht_4(mht_4_v, 262, "", "./tensorflow/compiler/xla/service/hlo_runner_interface.cc", "HloRunnerInterface::ReadModuleFromModuleBinaryProtofile");

  HloModuleProto module_proto;
  TF_RETURN_IF_ERROR(tensorflow::ReadBinaryProto(tensorflow::Env::Default(),
                                                 filename, &module_proto));

  TF_ASSIGN_OR_RETURN(
      HloModuleConfig module_config,
      HloModule::CreateModuleConfigFromProto(module_proto, debug_options));

  return HloModule::CreateFromProto(module_proto, module_config);
}

StatusOr<Literal> HloRunnerInterface::Execute(
    std::unique_ptr<HloModule> module, absl::Span<const Literal> arguments,
    bool run_hlo_passes, ExecutionProfile* profile) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc mht_5(mht_5_v, 279, "", "./tensorflow/compiler/xla/service/hlo_runner_interface.cc", "HloRunnerInterface::Execute");

  // Construct a vector of plain pointers for the arguments.
  std::vector<const Literal*> argument_pointers;
  argument_pointers.reserve(arguments.size());
  for (const auto& argument : arguments) {
    argument_pointers.push_back(&argument);
  }
  return Execute(
      /*module=*/std::move(module),
      /*arguments=*/argument_pointers,
      /*run_hlo_passes=*/run_hlo_passes,
      /*profile=*/profile);
}

StatusOr<Literal> HloRunnerInterface::ExecuteWithExecutable(
    Executable* executable, absl::Span<const Literal> arguments,
    ExecutionProfile* profile) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc mht_6(mht_6_v, 298, "", "./tensorflow/compiler/xla/service/hlo_runner_interface.cc", "HloRunnerInterface::ExecuteWithExecutable");

  // Construct a vector of plain pointers for the arguments.
  std::vector<const Literal*> argument_pointers;
  argument_pointers.reserve(arguments.size());
  for (const auto& argument : arguments) {
    argument_pointers.push_back(&argument);
  }
  return ExecuteWithExecutable(executable, argument_pointers, nullptr);
}

void HloRunnerInterface::UpdateEntryComputationLayout(
    HloModule* module, DeviceShapeRepresentationFn shape_representation_fn) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePShlo_runner_interfaceDTcc mht_7(mht_7_v, 312, "", "./tensorflow/compiler/xla/service/hlo_runner_interface.cc", "HloRunnerInterface::UpdateEntryComputationLayout");

  CHECK(shape_representation_fn != nullptr);
  // Make sure entry computation shapes are in device representation.
  for (int i = 0; i < module->entry_computation_layout().parameter_count();
       i++) {
    Shape shape =
        module->entry_computation_layout().parameter_layout(i).shape();
    *module->mutable_entry_computation_layout()->mutable_parameter_layout(i) =
        ShapeLayout(shape_representation_fn(shape));
  }
  *module->mutable_entry_computation_layout()->mutable_result_layout() =
      ShapeLayout(shape_representation_fn(
          module->entry_computation_layout().result_layout().shape()));
}

}  // namespace xla
