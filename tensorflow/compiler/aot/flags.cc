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
class MHTracer_DTPStensorflowPScompilerPSaotPSflagsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSaotPSflagsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSaotPSflagsDTcc() {
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

#include "tensorflow/compiler/aot/flags.h"

namespace tensorflow {
namespace tfcompile {

void AppendMainFlags(std::vector<Flag>* flag_list, MainFlags* flags) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSaotPSflagsDTcc mht_0(mht_0_v, 190, "", "./tensorflow/compiler/aot/flags.cc", "AppendMainFlags");

  const std::vector<Flag> tmp = {
      {"graph", &flags->graph,
       "Input GraphDef file.  If the file ends in '.pbtxt' it is expected to "
       "be in the human-readable proto text format, otherwise it is expected "
       "to be in the proto binary format."},
      {"debug_info", &flags->debug_info,
       "Graph debug info file.  If the file ends in '.pbtxt' it is expected to "
       "be in the human-readable proto text format, otherwise it is expected "
       "to be in the proto binary format."},
      {"debug_info_path_begin_marker", &flags->debug_info_path_begin_marker,
       "If not none, only keep the file path in the debug information after the"
       " marker. The default value is empty"},
      {"config", &flags->config,
       "Input file containing Config proto.  If the file ends in '.pbtxt' it "
       "is expected to be in the human-readable proto text format, otherwise "
       "it is expected to be in the proto binary format."},
      {"dump_fetch_nodes", &flags->dump_fetch_nodes,
       "If set, only flags related to fetches are processed, and the resulting "
       "fetch nodes will be dumped to stdout in a comma-separated list.  "
       "Typically used to format arguments for other tools, e.g. "
       "freeze_graph."},
      // Flags controlling the XLA ahead-of-time compilation, that correspond to
      // the fields of xla::cpu::CpuAotCompilationOptions.
      //
      // TODO(toddw): The following flags also need to be supported:
      //   --xla_cpu_llvm_opt_level
      //   --xla_cpu_llvm_cl_opts
      {"target_triple", &flags->target_triple,
       "Target platform, similar to the clang -target flag.  The general "
       "format is <arch><sub>-<vendor>-<sys>-<abi>.  "
       "http://clang.llvm.org/docs/CrossCompilation.html#target-triple."},
      {"target_cpu", &flags->target_cpu,
       "Target cpu, similar to the clang -mcpu flag.  "
       "http://clang.llvm.org/docs/CrossCompilation.html#cpu-fpu-abi"},
      {"target_features", &flags->target_features,
       "Target features, e.g. +avx2, +neon, etc."},
      {"entry_point", &flags->entry_point,
       "Name of the generated function.  If multiple generated object files "
       "will be linked into the same binary, each will need a unique entry "
       "point."},
      {"cpp_class", &flags->cpp_class,
       "Name of the generated C++ class, wrapping the generated function.  The "
       "syntax of this flag is [[<optional_namespace>::],...]<class_name>.  "
       "This mirrors the C++ syntax for referring to a class, where multiple "
       "namespaces may precede the class name, separated by double-colons.  "
       "The class will be generated in the given namespace(s), or if no "
       "namespaces are given, within the global namespace."},
      {"out_function_object", &flags->out_function_object,
       "Output object file containing the generated function for the "
       "TensorFlow model."},
      {"out_header", &flags->out_header, "Output header file name."},
      {"out_metadata_object", &flags->out_metadata_object,
       "Output object file name containing optional metadata for the generated "
       "function."},
      {"out_session_module", &flags->out_session_module,
       "Output session module proto."},
      {"mlir_components", &flags->mlir_components,
       "The MLIR components to enable. Currently only Bridge is supported."},
      {"experimental_quantize", &flags->experimental_quantize,
       "If set, quantization passes will run and dump the result before HLO "
       "code generation."},
      {"gen_name_to_index", &flags->gen_name_to_index,
       "Generate name-to-index data for Lookup{Arg,Result}Index methods."},
      {"gen_program_shape", &flags->gen_program_shape,
       "Generate program shape data for the ProgramShape method."},
  };
  flag_list->insert(flag_list->end(), tmp.begin(), tmp.end());
}

}  // namespace tfcompile
}  // namespace tensorflow
