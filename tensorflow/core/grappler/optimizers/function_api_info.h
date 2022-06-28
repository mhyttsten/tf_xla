/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_API_INFO_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_API_INFO_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTh() {
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


#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
class FunctionApiInfo {
 public:
  FunctionApiInfo();
  virtual ~FunctionApiInfo();

  enum FunctionType {
    INFERENCE,  // Default type.
    FORWARD,
    BACKWARD,
  };

  Status Init(const FunctionDef& function_def);

  const string& interface_name() const;
  const string& preferred_device() const;
  const FunctionType function_type() const;
  const string& pairing_function_name() const;
  const DataTypeVector& input_arg_dtypes() const;
  const DataTypeVector& output_arg_dtypes() const;

 private:
  string interface_name_;
  string preferred_device_;
  FunctionType function_type_;
  // The pairing function is used to pair between forward and backward function,
  // which will be useful during function swapping. Inference function won't
  // have pairing function.
  string pairing_function_name_;
  // The following two attributes are useful for forward and backward functions.
  DataTypeVector input_arg_dtypes_;
  DataTypeVector output_arg_dtypes_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionApiInfo);
};

// A collection of information for function and the interface it implements.
// A interface is a well defined math operation, eg I1 = 2 * x + y. Multiple
// functions could implement the same interface with different behavior based on
// different hardware condition and limits,
// eg F1 = math_ops.add(math_ops.add(x, x), y), or
//    F2 = math_ops.add(math_ops.matmul(x, 2), y).
class FunctionLibraryApiInfo {
 public:
  FunctionLibraryApiInfo();
  virtual ~FunctionLibraryApiInfo();
  // Populate the internal field for the functions within the function_library.
  Status Init(const FunctionDefLibrary& function_library);

  Status GetEquivalentImplementations(
      const string& function_name, std::vector<string>* other_functions) const;

  const FunctionApiInfo* GetApiInfo(const string& function_name) const;
  bool empty() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTh mht_0(mht_0_v, 252, "", "./tensorflow/core/grappler/optimizers/function_api_info.h", "empty");
 return func_info_.empty(); }
  std::size_t size() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_infoDTh mht_1(mht_1_v, 256, "", "./tensorflow/core/grappler/optimizers/function_api_info.h", "size");
 return func_info_.size(); }

 private:
  // Map between function name to function details.
  std::unordered_map<string, std::unique_ptr<FunctionApiInfo>> func_info_;

  // Map between interface name to function names.
  // Forward/backward function pair usually have different signatures between
  // each other since forward function could produce extra internal state as
  // output, and backward will take those extra state as inputs.
  absl::flat_hash_map<string, std::vector<string>> intf_to_inference_funcs_;
  absl::flat_hash_map<string, std::vector<string>> intf_to_forward_funcs_;
  absl::flat_hash_map<string, std::vector<string>> intf_to_backward_funcs_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionLibraryApiInfo);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_API_INFO_H_
