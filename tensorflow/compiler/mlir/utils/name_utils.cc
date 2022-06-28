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
class MHTracer_DTPStensorflowPScompilerPSmlirPSutilsPSname_utilsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSutilsPSname_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSutilsPSname_utilsDTcc() {
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

#include "tensorflow/compiler/mlir/utils/name_utils.h"

#include <cctype>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project

namespace mlir {

namespace {
// Checks if a character is legal for a TensorFlow node name, with special
// handling if a character is at the beginning.
bool IsLegalChar(char c, bool first_char) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScompilerPSmlirPSutilsPSname_utilsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/mlir/utils/name_utils.cc", "IsLegalChar");

  if (isalpha(c)) return true;
  if (isdigit(c)) return true;
  if (c == '.') return true;
  if (c == '_') return true;

  // First character of a node name can only be a letter, digit, dot or
  // underscore.
  if (first_char) return false;

  if (c == '/') return true;
  if (c == '-') return true;

  return false;
}
}  // anonymous namespace

void LegalizeNodeName(std::string& name) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSutilsPSname_utilsDTcc mht_1(mht_1_v, 220, "", "./tensorflow/compiler/mlir/utils/name_utils.cc", "LegalizeNodeName");

  if (name.empty()) return;

  if (!IsLegalChar(name[0], /*first_char=*/true)) name[0] = '.';

  for (char& c : llvm::drop_begin(name, 1))
    if (!IsLegalChar(c, /*first_char=*/false)) c = '.';
}

std::string GetNameFromLoc(Location loc) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSutilsPSname_utilsDTcc mht_2(mht_2_v, 232, "", "./tensorflow/compiler/mlir/utils/name_utils.cc", "GetNameFromLoc");

  llvm::SmallVector<llvm::StringRef, 8> loc_names;
  llvm::SmallVector<Location, 8> locs;
  locs.push_back(loc);
  bool names_is_nonempty = false;

  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();

    if (auto name_loc = curr_loc.dyn_cast<NameLoc>()) {
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      auto name = name_loc.getName().strref().split('@').first;
      // Skip if the name is for op type.
      if (!name.endswith(":")) {
        loc_names.push_back(name);
        if (!name.empty()) names_is_nonempty = true;
      }
      continue;
    } else if (auto call_loc = curr_loc.dyn_cast<CallSiteLoc>()) {
      // Use location of the Callee to generate the name.
      locs.push_back(call_loc.getCallee());
      continue;
    } else if (auto fused_loc = curr_loc.dyn_cast<FusedLoc>()) {
      // Push all locations in FusedLoc in reverse order, so locations are
      // visited based on order in FusedLoc.
      auto reversed_fused_locs = llvm::reverse(fused_loc.getLocations());
      locs.append(reversed_fused_locs.begin(), reversed_fused_locs.end());
      continue;
    }

    // Location is not a supported, so an empty StringRef is added.
    loc_names.push_back(llvm::StringRef());
  }

  if (names_is_nonempty)
    return llvm::join(loc_names.begin(), loc_names.end(), ";");

  return "";
}

std::string GetOpTypeFromLoc(Location loc) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSutilsPSname_utilsDTcc mht_3(mht_3_v, 276, "", "./tensorflow/compiler/mlir/utils/name_utils.cc", "GetOpTypeFromLoc");

  llvm::SmallVector<llvm::StringRef, 1> loc_op_types;
  llvm::SmallVector<Location, 8> locs;
  locs.push_back(loc);
  bool op_types_is_nonempty = false;

  while (!locs.empty()) {
    Location curr_loc = locs.pop_back_val();

    if (auto name_loc = curr_loc.dyn_cast<NameLoc>()) {
      // Add name in NameLoc. For NameLoc we also account for names due to ops
      // in functions where the op's name is first.
      auto op_type = name_loc.getName().strref().split('@').first;
      if (op_type.endswith(":")) {
        op_type = op_type.substr(0, op_type.size() - 1);
        loc_op_types.push_back(op_type);
        if (!op_type.empty()) op_types_is_nonempty = true;
      }
      continue;
    } else if (auto call_loc = curr_loc.dyn_cast<CallSiteLoc>()) {
      // Use location of the Callee to generate the name.
      locs.push_back(call_loc.getCallee());
      continue;
    } else if (auto fused_loc = curr_loc.dyn_cast<FusedLoc>()) {
      // The first location is reserved for op_type.
      if (!fused_loc.getLocations().empty())
        locs.push_back(fused_loc.getLocations()[0]);
      continue;
    }

    // Location is not a supported, so an empty StringRef is added.
    loc_op_types.push_back(llvm::StringRef());
  }

  if (op_types_is_nonempty)
    return llvm::join(loc_op_types.begin(), loc_op_types.end(), ";");

  return "";
}

}  // namespace mlir
