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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_supported_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_supported_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_supported_opsDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/tf2xla_supported_ops.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace tf2xla {
namespace {

void PrintSupportedOps(const string& device, const string& regen_run) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device: \"" + device + "\"");
   mht_0_v.push_back("regen_run: \"" + regen_run + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_supported_opsDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/tf2xla/tf2xla_supported_ops.cc", "PrintSupportedOps");

  XlaOpRegistry::RegisterCompilationKernels();

  std::vector<const KernelDef*> kdefs =
      XlaOpRegistry::DeviceKernels(device,
                                   /*include_compilation_only_kernels=*/true);
  std::sort(
      kdefs.begin(), kdefs.end(),
      [](const KernelDef* a, const KernelDef* b) { return a->op() < b->op(); });

  std::cout << "**Supported operators for device: " << device << "**\n\n"
            << "Operator | Type Constraint\n"
            << "-------- | ---------------" << std::endl;
  for (const KernelDef* kdef : kdefs) {
    std::vector<string> constraints;
    constraints.reserve(kdef->constraint().size());
    for (const KernelDef::AttrConstraint& constraint : kdef->constraint()) {
      std::vector<string> types;
      const auto& allowed_values = constraint.allowed_values().list().type();
      types.reserve(allowed_values.size());
      for (int type : allowed_values) {
        types.push_back(DataTypeString(static_cast<DataType>(type)));
      }
      std::sort(types.begin(), types.end());
      constraints.push_back("`" + constraint.name() + "={" +
                            absl::StrJoin(types, ",") + "}`");
    }
    std::cout << "`" << kdef->op() << "` | "
              << absl::StrJoin(constraints, "<br>") << std::endl;
  }

  std::cout << "\nTo regenerate this table, run:\n\n```shell\n"
            << regen_run << " --device=" << device << "\n```" << std::endl;
}

}  // namespace

void SupportedOpsMain(int argc, char** argv, const char* regen_run) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("regen_run: \"" + (regen_run == nullptr ? std::string("nullptr") : std::string((char*)regen_run)) + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPStf2xla_supported_opsDTcc mht_1(mht_1_v, 247, "", "./tensorflow/compiler/tf2xla/tf2xla_supported_ops.cc", "SupportedOpsMain");

  std::vector<string> device_names = XlaOpRegistry::BackendNames();
  std::sort(device_names.begin(), device_names.end());

  // Set up and parse flags.
  string device;
  std::vector<Flag> flag_list = {
      {"device", &device,
       "Name of the compilation device for which to print supported ops, "
       "one of: " +
           absl::StrJoin(device_names, ",")},
  };
  string usage = Flags::Usage(argv[0], flag_list);
  bool parsed_flags_ok = Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;
  QCHECK(XlaOpRegistry::IsBackendRegistered(device))
      << "\nUnknown device: " << device << "\n"
      << usage;

  // Run the program.
  port::InitMain(usage.c_str(), &argc, &argv);
  QCHECK(argc == 1) << "\nERROR: This command does not take any arguments "
                       "other than flags\n\n"
                    << usage;
  PrintSupportedOps(device, regen_run);
}

}  // namespace tf2xla
}  // namespace tensorflow
