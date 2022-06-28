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
class MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc() {
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

#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/utils/name_utils.h"

static inline absl::string_view StringRefToView(llvm::StringRef ref) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "StringRefToView");

  return absl::string_view(ref.data(), ref.size());
}

static inline llvm::StringRef StringViewToRef(absl::string_view view) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("view: \"" + std::string(view.data(), view.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "StringViewToRef");

  return llvm::StringRef(view.data(), view.size());
}

namespace tensorflow {

OpOrArgNameMapper::~OpOrArgNameMapper() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_2(mht_2_v, 219, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "OpOrArgNameMapper::~OpOrArgNameMapper");
}

llvm::StringRef OpOrArgNameMapper::GetUniqueName(llvm::StringRef prefix) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_3(mht_3_v, 224, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "OpOrArgNameMapper::GetUniqueName");

  // Insert/find if prefix is unique.
  auto prefix_it = name_to_count_.try_emplace(prefix, 0);
  if (prefix_it.second && IsUnique(prefix)) {
    // Name is unique, increment count and return string name backed by
    // `name_to_count_`.
    ++prefix_it.first->second;
    return prefix_it.first->first();
  }

  // Add increasing number (count) to end of prefix until it is determined
  // to be unique.
  auto& val = prefix_it.first->second;
  llvm::SmallString<64> probe_name(prefix);
  probe_name.append(GetSuffixSeparator());
  const int probe_prefix_size = probe_name.size();
  while (true) {
    probe_name.resize(probe_prefix_size);
    // TODO(jpienaar): Subtract one so that the initial suffix is 0 instead
    // of 1.
    // TODO(jpienaar): Switch to radix 36 and update tests.
    llvm::APInt(32, val++).toString(probe_name, /*Radix=*/10, /*Signed=*/false);
    if (IsUnique(probe_name)) {
      // Insert/find if prefix with appended number is unique.
      auto probe_name_it = name_to_count_.try_emplace(probe_name, 1);
      if (probe_name_it.second) {
        // Name is unique, return string name backed by `name_to_count_`.
        return probe_name_it.first->first();
      }
    }
  }
}

llvm::StringRef OpOrArgNameMapper::GetUniqueName(OpOrVal op_or_val) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_4(mht_4_v, 260, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "OpOrArgNameMapper::GetUniqueName");

  auto& name = op_or_val_to_name_[op_or_val];
  if (!name.empty()) return StringViewToRef(name);
  // Update the value in the map with unique name.
  llvm::StringRef ref = GetUniqueName(GetName(op_or_val));
  name = StringRefToView(ref);
  return ref;
}

absl::string_view OpOrArgNameMapper::GetUniqueNameView(OpOrVal op_or_val) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_5(mht_5_v, 272, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "OpOrArgNameMapper::GetUniqueNameView");

  auto& name = op_or_val_to_name_[op_or_val];
  if (!name.empty()) return name;
  // Update the value in the map with unique name.
  name = StringRefToView(GetUniqueName(GetName(op_or_val)));
  return name;
}

int OpOrArgNameMapper::InitOpName(OpOrVal op_or_val, llvm::StringRef name) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_6(mht_6_v, 283, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "OpOrArgNameMapper::InitOpName");

  auto it = name_to_count_.try_emplace(name, 0);
  auto inserted = op_or_val_to_name_.try_emplace(
      op_or_val, StringRefToView(it.first->first()));
  (void)inserted;
  // TODO(jpienaar): Debug cases where we expect this behavior.
  // assert(inserted.second && "op_or_val already initialized");
  return it.first->second++;
}

bool OpOrArgNameMapper::IsUnique(llvm::StringRef name) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_7(mht_7_v, 296, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "OpOrArgNameMapper::IsUnique");
 return true; }

std::string OpOrArgLocNameMapper::GetName(OpOrVal op_or_val) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_8(mht_8_v, 301, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "OpOrArgLocNameMapper::GetName");

  if (auto* op = op_or_val.dyn_cast<mlir::Operation*>()) {
    auto name_from_loc = mlir::GetNameFromLoc(op->getLoc());
    if (!name_from_loc.empty()) return name_from_loc;
    // If the location is none of the expected types, then simply use name
    // generated using the op type.
    return std::string(op->getName().getStringRef());
  }
  auto val = op_or_val.dyn_cast<mlir::Value>();
  auto name_from_loc = mlir::GetNameFromLoc(val.getLoc());
  if (!name_from_loc.empty()) return name_from_loc;
  // If the location is none of the expected types, then simply use name
  // generated using the op type. Follow TF convention and append the result
  // index unless 0.
  if (auto result = val.dyn_cast<mlir::OpResult>()) {
    if (result.getResultNumber() > 0)
      return llvm::formatv("{0}:{1}",
                           result.getOwner()->getName().getStringRef(),
                           result.getResultNumber());
    return std::string(result.getOwner()->getName().getStringRef());
  }
  // Use the ASM syntax for BlockArgument
  if (auto arg = val.dyn_cast<mlir::BlockArgument>()) {
    return "arg" + std::to_string(arg.getArgNumber());
  }
  return "";
}

std::string OpOrArgStripNameMapper::GetName(OpOrVal op_or_val) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTcc mht_9(mht_9_v, 332, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.cc", "OpOrArgStripNameMapper::GetName");

  return llvm::toString(llvm::APInt(32, count_++),
                        /*Radix=*/36, /*Signed=*/false);
}

}  // namespace tensorflow
