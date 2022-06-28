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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_PARAMETER_BINDING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_PARAMETER_BINDING_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTh() {
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


#include <functional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

class HloModule;
// We currently use an explicit API that takes an extra parameter to indicate
// the runtime size of a dynamic dimension. DynamicParameterBinding indicates
// the relationship between parameter: We can have a dynamic parameter that
// points to another target parameter to indicate that the target parameter is
// dynamic.
//
//
// TODO(b/119520625): Remove this API once we have more dynamic shape infra
// ready.
class DynamicParameterBinding {
 public:
  // DynamicParameter represents a special parameter that is used to represent
  // the runtime size of a dimension of another parameter. A dynamic parameter
  // has to be a scalar value.
  struct DynamicParameter {
    // The parameter number of dynamic parameter.
    int64_t parameter_num;
    // The index of the parameter.
    ShapeIndex parameter_index;
  };

  // DynamicDimension represents a dimension whose size is determined at
  // runtime. A DynamicDimension's runtime size is determined by the binded
  // DynamicParameter using `DynamicParameterBinding::Bind` method.
  struct DynamicDimension {
    // The parameter number of dynamic dimension.
    int64_t parameter_num;
    // The subshape index of the parameter.
    ShapeIndex parameter_index;
    // The dimension number in the subshape.
    int64_t dimension;

    // "friend" keyword are added so these functions can be found by ADL.
    template <typename H>
    friend H AbslHashValue(H h, const DynamicDimension& m) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSdynamic_parameter_bindingDTh mht_0(mht_0_v, 235, "", "./tensorflow/compiler/xla/service/dynamic_parameter_binding.h", "AbslHashValue");

      return H::combine(std::move(h), m.parameter_num, m.parameter_index,
                        m.dimension);
    }

    friend bool operator==(const DynamicDimension& lhs,
                           const DynamicDimension& rhs) {
      return lhs.parameter_num == rhs.parameter_num &&
             lhs.parameter_index == rhs.parameter_index &&
             lhs.dimension == rhs.dimension;
    }
  };

  DynamicParameterBinding() = default;

  virtual ~DynamicParameterBinding() = default;

  // Adds binding which indicates that the dimension indicated by
  // `dynamic_dimension` is dynamic, and its runtime size is represented by
  // `dynamic_parameter`.
  Status Bind(const DynamicParameter& dynamic_parameter,
              const DynamicDimension& dynamic_dimension);

  // Returns the parameter and the index representing the runtime size of
  // dimension `dim_num` of parameter `param_num` at `param_index`.
  //
  // Returns nullopt if the binding is not set.
  absl::optional<DynamicParameter> GetBinding(
      const DynamicDimension& dynamic_dimension) const;

  using BindingFn =
      std::function<Status(const DynamicParameter& dynamic_parameter,
                           const DynamicDimension& dynamic_dimension)>;

  // Iterate through each binding.
  Status ForEachBinding(BindingFn fn) const;

  DynamicParameterBindingProto ToProto() const;

  static StatusOr<DynamicParameterBinding> CreateFromProto(
      const DynamicParameterBindingProto& proto);

  std::string ToString() const;

  // Verifies that the given binding is valid for the given module.
  // Specifically, the binding's parameter and parameter size should be valid.
  Status Verify(const HloModule& module) const;

 private:
  // Keeps track of mappings from DynamicDimension to DynamicParameter. The
  // direction of is chosen so that we can easily query if a dimension is
  // dynamic and which dynamic parameter represents the real size of that
  // dimension.
  absl::flat_hash_map<DynamicDimension, DynamicParameter> bindings_;
};

std::ostream& operator<<(std::ostream& out,
                         const DynamicParameterBinding& binding);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DYNAMIC_PARAMETER_BINDING_H_
