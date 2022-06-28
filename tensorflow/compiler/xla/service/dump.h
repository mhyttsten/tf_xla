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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DUMP_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DUMP_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTh() {
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


#include "absl/strings/string_view.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla.pb.h"

// Consolidated utilities for logging information during compilation, usually
// based on the options specified in the DebugOptions proto.
//
// Most functions here take an HloModule and read the DebugOptions from the
// module's config.

namespace xla {

class BufferAssignment;
class HloExecutionProfile;
class HloSnapshot;

// Get a timestamp which we can use as a filename prefix specific to this
// module.
std::string TimestampFor(const HloModule& module);

// Create the filename we will use to dump in DumpToFileInDir.
std::string FilenameFor(const HloModule& module, absl::string_view prefix,
                        absl::string_view suffix);

// Writes the given string to a file in the xla_dump_to directory specified by
// module's DebugOptions.
//
// If module doesn't have an xla_dump_to directory, does nothing.
void DumpToFileInDir(const HloModule& module, absl::string_view file_prefix,
                     absl::string_view file_suffix, absl::string_view contents);
void DumpToFileInDir(const DebugOptions& debug_options,
                     absl::string_view filename, absl::string_view contents);

// Like DumpToFileInDir, except if module doesn't have an xla_dump_to directory
// specified, or if that directory is equal to "-", writes to stdout instead.
void DumpToFileInDirOrStdout(const HloModule& module,
                             absl::string_view file_prefix,
                             absl::string_view file_suffix,
                             absl::string_view contents);

// Like DumpToFileInDir, except if debug_options doesn't have an xla_dump_to
// directory specified, or if that directory is equal to "-", writes to stdout
// instead.
void DumpToFileInDirOrStdout(const DebugOptions& debug_options, int unique_id,
                             absl::string_view module_name,
                             absl::string_view file_prefix,
                             absl::string_view file_suffix,
                             absl::string_view contents);

// Writes the given op to a file in the xla_dump_to directory specified by
// module's DebugOptions. Sets the op's source locations to that file.
//
// If module doesn't have an xla_dump_to directory, does nothing.
void DumpToFileInDirOrStdout(const HloModule& module,
                             absl::string_view file_prefix,
                             mlir::Operation* op);

// Dumps the given protobuf to the given filename if dumping is enabled.
// Exactly where and in what formats it's dumped is determined by the debug
// options.
void DumpProtobufToFile(const tensorflow::protobuf::Message& proto,
                        const DebugOptions& debug_options,
                        absl::string_view filename);

// Similar to above, but the filename depends on module's information and the
// given name.
void DumpPerModuleProtobufToFile(const HloModule& module,
                                 const tensorflow::protobuf::Message& proto,
                                 const DebugOptions& debug_options,
                                 absl::string_view name);

// Dumps the given HLO module if dumping is enabled for the module. Exactly
// where and in what formats it's dumped is determined by the module's config.
//
// If you pass an HloExecutionProfile, note that currently only DOT-based output
// formats (i.e. --xla_dump_as_{dot,html,url}) are able to incorporate it into
// their output.  Other formats will just ignore the profile.
void DumpHloModuleIfEnabled(const HloModule& module, absl::string_view name);
void DumpHloModuleIfEnabled(const HloModule& module,
                            const BufferAssignment& buffer_assn,
                            absl::string_view name);
void DumpHloModuleIfEnabled(const HloModule& module,
                            const HloExecutionProfile& profile,
                            absl::string_view name);

// Dumps the given HLO module after running one HLO pass and before running
// another, if that's enabled. Returns the full file paths of all dumps of the
// module, or an empty vector if nothing was dumped.
std::vector<std::string> DumpHloModuleBetweenPassesIfEnabled(
    absl::string_view pipeline_name, absl::string_view before_pass_name,
    absl::string_view after_pass_name, const HloModule& module);

// Dumps the given HLO module during the given HLO pass, if that's enabled.
//
// "step" is a human-readable description of where we are in the middle of this
// pass.  For example, "before-assigning-layouts".
void DumpHloModuleDuringPassIfEnabled(absl::string_view pass_name,
                                      absl::string_view step,
                                      const HloModule& module);

// Dumps the given HloSnapshot to the module's xla_dump_dir, if this is enabled.
//
// Prefer the first overload below, as this will give filenames that are
// consistent with the other methods here.  The second overload (which doesn't
// take an HloModule) is useful in the cases when you're dumping an HloSnapshot
// and simply don't have an HloModule.
void DumpHloSnapshotIfEnabled(const HloModule& module,
                              const HloSnapshot& snapshot);
void DumpHloSnapshotIfEnabled(const HloSnapshot& snapshot,
                              const DebugOptions& opts);

void DumpHloModuleMetadataIfEnabled(const std::vector<HloModule*>& modules);

// Returns true if we should dump data for an HloModule.  This is useful if you
// want to check if DumpToFileInDir{,OrStdout} will do anything before
// generating an expensive string.
bool DumpingEnabledForHloModule(absl::string_view hlo_module_name,
                                const DebugOptions& opts);
inline bool DumpingEnabledForHloModule(const HloModule& module) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdumpDTh mht_0(mht_0_v, 308, "", "./tensorflow/compiler/xla/service/dump.h", "DumpingEnabledForHloModule");

  return DumpingEnabledForHloModule(module.name(),
                                    module.config().debug_options());
}

// Returns true if DumpToFileInDirOrStdout and DumpHloModuleIfEnabled will write
// to stdout, rather than to a file on disk.
//
// This is useful if you want to do something different when writing to stdout.
// For example, maybe you have (almost-)duplicate data that you wouldn't mind
// writing to two files, but you don't want to print twice.
bool DumpingToStdout(const DebugOptions& opts);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DUMP_H_
