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
class MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_module_loaderDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_module_loaderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_module_loaderDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Emits an HLO module in a text form suitable for diffing.

#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"

namespace xla {
namespace {

Status OverrideConfig(const hlo_module_loader_details::Config& ovr_config,
                      HloModuleConfig* config) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_module_loaderDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/tools/hlo_module_loader.cc", "OverrideConfig");

  config->set_replica_count(ovr_config.num_replicas);
  config->set_num_partitions(ovr_config.num_partitions);
  return Status::OK();
}

}  // namespace

std::string StripLogHeaders(const std::string& hlo_string) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("hlo_string: \"" + hlo_string + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPShlo_module_loaderDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/xla/tools/hlo_module_loader.cc", "StripLogHeaders");

  // I0521 12:04:45.883483    1509 service.cc:186] ...
  static RE2* matcher = new RE2(
      "[IWEF]\\d{4} "
      "\\d{2}:\\d{2}:\\d{2}\\.\\d+\\s+\\d+\\s+[^:]+:\\d+\\]\\s?(.*)");
  absl::string_view matches[4];
  std::vector<std::string> lines = absl::StrSplit(hlo_string, '\n');
  for (auto& line : lines) {
    if (matcher->Match(line, 0, line.size(), RE2::ANCHOR_START, matches, 4)) {
      line = std::string(matches[1]);
    }
  }
  return absl::StrJoin(lines, "\n",
                       [](std::string* out, const std::string& line) {
                         absl::StrAppend(out, line);
                       });
}

StatusOr<std::unique_ptr<HloModule>> LoadModuleFromData(
    const std::string& data, const std::string& format,
    hlo_module_loader_details::Config ovr_config,
    const std::function<void(HloModuleConfig*)>& config_modifier_hook) {
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  std::unique_ptr<HloModule> module;
  if (format == "hlo" || format == "txt") {
    std::string hlo_string = StripLogHeaders(data);
    HloModuleConfig config;
    config.set_debug_options(debug_options);
    TF_RETURN_IF_ERROR(OverrideConfig(ovr_config, &config));
    if (config_modifier_hook) {
      config_modifier_hook(&config);
    }
    TF_ASSIGN_OR_RETURN(module,
                        ParseAndReturnUnverifiedModule(hlo_string, config));
  } else {
    HloSnapshot proto;
    if (format == "pb") {
      if (!proto.ParseFromString(data) &&
          !proto.mutable_hlo()->ParseFromString(data) &&
          !proto.mutable_hlo()->mutable_hlo_module()->ParseFromString(data)) {
        return InvalidArgument("Failed to parse input as HLO protobuf binary");
      }
    } else if (format == "pbtxt") {
      if (!tensorflow::protobuf::TextFormat::ParseFromString(data, &proto) &&
          !tensorflow::protobuf::TextFormat::ParseFromString(
              data, proto.mutable_hlo()) &&
          !tensorflow::protobuf::TextFormat::ParseFromString(
              data, proto.mutable_hlo()->mutable_hlo_module())) {
        return InvalidArgument("Failed to parse input as HLO protobuf text");
      }
    } else {
      return InvalidArgument(
          "Invalid format from file extension: '%s'. Expected: hlo, txt, pb, "
          "or pbtxt",
          format);
    }
    TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                        HloModule::CreateModuleConfigFromProto(
                            proto.hlo().hlo_module(), debug_options));
    TF_RETURN_IF_ERROR(OverrideConfig(ovr_config, &config));
    if (config_modifier_hook) {
      config_modifier_hook(&config);
    }
    TF_ASSIGN_OR_RETURN(
        module, HloModule::CreateFromProto(proto.hlo().hlo_module(), config));
  }
  return std::move(module);
}

StatusOr<std::unique_ptr<HloModule>> LoadModuleFromFile(
    const std::string& path, hlo_module_loader_details::Config ovr_config,
    std::string format,
    const std::function<void(HloModuleConfig*)>& config_modifier_hook) {
  std::string data;
  if (format.empty()) {
    format = std::string(tensorflow::io::Extension(path));
  }
  TF_RETURN_IF_ERROR(
      tensorflow::ReadFileToString(tensorflow::Env::Default(), path, &data));
  return LoadModuleFromData(data, format, ovr_config, config_modifier_hook);
}

}  // namespace xla
