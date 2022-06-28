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

#ifndef TENSORFLOW_COMPILER_MLIR_op_or_val_NAME_MAPPER_H_
#define TENSORFLOW_COMPILER_MLIR_op_or_val_NAME_MAPPER_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTh() {
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


#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project

namespace tensorflow {

// PointerUnion for operation and value.
// TODO(jpienaar): Rename the files.
using OpOrVal = llvm::PointerUnion<mlir::Operation*, mlir::Value>;

// Mapper from operation or value to name.
class OpOrArgNameMapper {
 public:
  // Returns unique name for the given prefix.
  llvm::StringRef GetUniqueName(llvm::StringRef prefix);

  // Returns unique name for the operation or value.
  llvm::StringRef GetUniqueName(OpOrVal op_or_val);

  // Returns unique name as a string_view for the operation or value.
  absl::string_view GetUniqueNameView(OpOrVal op_or_val);

  // Initializes operation or value to map to name. Returns number of
  // operations or value already named 'name' which should be 0 else
  // GetUniqueName could return the same names for different operations or
  // values.
  // Note: Its up to the caller to decide the behavior when assigning two
  // operations or values to the same name.
  int InitOpName(OpOrVal op_or_val, llvm::StringRef name);

  virtual ~OpOrArgNameMapper();

 protected:
  // Returns true if the name is unique. A derived class can override it if the
  // class maintains uniqueness in a different scope.
  virtual bool IsUnique(llvm::StringRef name);

  // Returns a constant view of the underlying map.
  const llvm::DenseMap<OpOrVal, absl::string_view>& GetMap() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTh mht_0(mht_0_v, 232, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.h", "GetMap");

    return op_or_val_to_name_;
  }

  // Returns the separator used before uniqueing suffix.
  virtual llvm::StringRef GetSuffixSeparator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSop_or_arg_name_mapperDTh mht_1(mht_1_v, 240, "", "./tensorflow/compiler/mlir/op_or_arg_name_mapper.h", "GetSuffixSeparator");
 return ""; }

 private:
  // Returns name from the location of the operation or value.
  virtual std::string GetName(OpOrVal op_or_val) = 0;

  // Maps string name to count. This map is used to help keep track of unique
  // names for operations or values.
  llvm::StringMap<int64_t> name_to_count_;
  // Maps operation or values to name. Value in map is a view of the string
  // name in `name_to_count_`. Names in `name_to_count_` are never removed.
  llvm::DenseMap<OpOrVal, absl::string_view> op_or_val_to_name_;
};

// OpOrArgNameMapper that returns, for operations or values not initialized
// to a specific name, a name based on the location of the operation or
// value.
class OpOrArgLocNameMapper : public OpOrArgNameMapper {
 protected:
  std::string GetName(OpOrVal op_or_val) override;
};

// OpOrArgNameMapper that returns, for operations or values not initialized
// to a specific name, a short name.
class OpOrArgStripNameMapper : public OpOrArgNameMapper {
 private:
  std::string GetName(OpOrVal op_or_val) override;

  // Number of ops mapped.
  int count_ = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_op_or_val_NAME_MAPPER_H_
