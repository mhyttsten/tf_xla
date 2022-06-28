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
class MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSdumped_computation_to_operation_listDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSdumped_computation_to_operation_listDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSdumped_computation_to_operation_listDTcc() {
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

// Dumps out the operations that are present in a serialized computation.

#include <iostream>
#include <memory>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace tools {

class OperationDumper : public DfsHloVisitorWithDefault {
 public:
  explicit OperationDumper(const std::string& path) : path_(path) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("path: \"" + path + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSdumped_computation_to_operation_listDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/xla/tools/dumped_computation_to_operation_list.cc", "OperationDumper");
}

  Status DefaultAction(HloInstruction* hlo) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSdumped_computation_to_operation_listDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/tools/dumped_computation_to_operation_list.cc", "DefaultAction");

    std::string params = absl::StrJoin(
        hlo->operands(), ", ",
        [](std::string* out, const HloInstruction* operand) {
          absl::StrAppend(out, ShapeUtil::HumanString(operand->shape()));
        });
    // Spit `op_name(params...) -> result_type :: path` to stdout.
    std::cout << absl::StrFormat("%s :: (%s) -> %s :: %s\n",
                                 HloOpcodeString(hlo->opcode()), params,
                                 ShapeUtil::HumanString(hlo->shape()), path_);
    return Status::OK();
  }

 private:
  std::string path_;
};

void RealMain(absl::Span<char* const> args) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSdumped_computation_to_operation_listDTcc mht_2(mht_2_v, 238, "", "./tensorflow/compiler/xla/tools/dumped_computation_to_operation_list.cc", "RealMain");

  LocalClient* client = ClientLibrary::LocalClientOrDie();
  LocalService* local_service =
      ClientLibrary::GetXlaService(client->platform());
  for (char* arg : args) {
    HloSnapshot snapshot;
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), arg,
                                            &snapshot));
    auto computation_status = client->LoadSnapshot(snapshot);
    if (!computation_status.ok()) {
      fprintf(stderr, "could not load snapshot for %s: %s\n", arg,
              computation_status.status().ToString().c_str());
      continue;
    }
    XlaComputation computation = computation_status.ConsumeValueOrDie();

    std::unique_ptr<ProgramShape> program_shape =
        client->GetComputationShape(computation).ConsumeValueOrDie();

    std::vector<const Shape*> layouts;
    layouts.reserve(program_shape->parameters_size());
    for (int i = 0; i < program_shape->parameters_size(); ++i) {
      layouts.push_back(&program_shape->parameters(i));
    }
    ExecutableBuildOptions build_options;
    build_options.set_device_ordinal(0);
    build_options.set_result_layout(program_shape->result());
    auto executables =
        local_service->CompileExecutables(computation, layouts, build_options)
            .ConsumeValueOrDie();
    CHECK_EQ(executables.size(), 1);
    const HloModule& module = executables[0]->module();

    OperationDumper dumper(arg);
    for (auto* computation : module.computations()) {
      TF_CHECK_OK(computation->Accept(&dumper));
    }
  }
}

}  // namespace tools
}  // namespace xla

int main(int argc, char** argv) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPStoolsPSdumped_computation_to_operation_listDTcc mht_3(mht_3_v, 284, "", "./tensorflow/compiler/xla/tools/dumped_computation_to_operation_list.cc", "main");

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  absl::Span<char* const> args(argv, argc);
  args.remove_prefix(1);  // Pop off the binary name, argv[0]
  xla::tools::RealMain(args);
  return 0;
}
