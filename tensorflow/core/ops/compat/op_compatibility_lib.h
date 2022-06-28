/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_OPS_COMPAT_OP_COMPATIBILITY_LIB_H_
#define TENSORFLOW_CORE_OPS_COMPAT_OP_COMPATIBILITY_LIB_H_
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
class MHTracer_DTPStensorflowPScorePSopsPScompatPSop_compatibility_libDTh {
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
   MHTracer_DTPStensorflowPScorePSopsPScompatPSop_compatibility_libDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPScompatPSop_compatibility_libDTh() {
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


#include <set>

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class OpCompatibilityLib {
 public:
  // `ops_prefix` is a filename prefix indicating where to find the
  //   ops files.
  // `history_version` is used to construct the ops history file name.
  // `*stable_ops` has an optional list of ops that we care about.
  //   If stable_ops == nullptr, we use all registered ops.
  //   Otherwise ValidateCompatible() ignores ops not in *stable_ops
  //   and require all ops in *stable_ops to exist.
  OpCompatibilityLib(const string& ops_prefix, const string& history_version,
                     const std::set<string>* stable_ops);

  // Name of the file that contains the checked-in versions of *all*
  // ops, with docs.
  const string& ops_file() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPScompatPSop_compatibility_libDTh mht_0(mht_0_v, 210, "", "./tensorflow/core/ops/compat/op_compatibility_lib.h", "ops_file");
 return ops_file_; }

  // Name of the file that contains all versions of *stable* ops,
  // without docs.  Op history is in (alphabetical, oldest-first)
  // order.
  const string& op_history_file() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSopsPScompatPSop_compatibility_libDTh mht_1(mht_1_v, 218, "", "./tensorflow/core/ops/compat/op_compatibility_lib.h", "op_history_file");
 return op_history_file_; }

  // Name of the directory that contains all versions of *stable* ops,
  // without docs.  Op history is one file per op, in oldest-first
  // order within the file.
  const string& op_history_directory() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSopsPScompatPSop_compatibility_libDTh mht_2(mht_2_v, 226, "", "./tensorflow/core/ops/compat/op_compatibility_lib.h", "op_history_directory");
 return op_history_directory_; }

  // Should match the contents of ops_file().  Run before calling
  // ValidateCompatible().
  string OpsString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSopsPScompatPSop_compatibility_libDTh mht_3(mht_3_v, 233, "", "./tensorflow/core/ops/compat/op_compatibility_lib.h", "OpsString");
 return op_list_.DebugString(); }

  // Returns the number of ops in OpsString(), includes all ops, not
  // just stable ops.
  int num_all_ops() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSopsPScompatPSop_compatibility_libDTh mht_4(mht_4_v, 240, "", "./tensorflow/core/ops/compat/op_compatibility_lib.h", "num_all_ops");
 return op_list_.op_size(); }

  // <file name, file contents> pairs representing op history.
  typedef std::vector<std::pair<string, OpList>> OpHistory;

  // Make sure the current version of the *stable* ops are compatible
  // with the historical versions, and if out_op_history != nullptr,
  // generate a new history adding all changed ops.  Sets
  // *changed_ops/*added_ops to the number of changed/added ops
  // (ignoring doc changes).
  Status ValidateCompatible(Env* env, int* changed_ops, int* added_ops,
                            OpHistory* out_op_history);

 private:
  const string ops_file_;
  const string op_history_file_;
  const string op_history_directory_;
  const std::set<string>* stable_ops_;
  OpList op_list_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_OPS_COMPAT_OP_COMPATIBILITY_LIB_H_
