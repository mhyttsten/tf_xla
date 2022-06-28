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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_LAYOUT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_LAYOUT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh() {
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
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Class which contains the layouts of the parameters and results of a
// computation. The layouts are stored as ShapeLayouts with immutable shapes and
// mutable layouts.
class ComputationLayout {
 public:
  // Creates a new ComputationLayout with the given result layout.
  explicit ComputationLayout(ShapeLayout result_layout)
      : result_layout_(std::move(result_layout)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/computation_layout.h", "ComputationLayout");
}

  // Constructs a ComputationLayout from a ProgramShape. The layouts of the
  // parameters and results are set to the default layout. Layouts in the
  // ProgramShape are ignored if ignore_layouts is true.
  explicit ComputationLayout(const ProgramShape& program_shape,
                             bool ignore_layouts = true);

  // Adds a new parameter layout to the computation layout.
  void add_parameter_layout(ShapeLayout shape_layout) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_1(mht_1_v, 218, "", "./tensorflow/compiler/xla/service/computation_layout.h", "add_parameter_layout");

    parameter_layouts_.push_back(std::move(shape_layout));
  }

  // Returns the layout of a particular parameter.
  const ShapeLayout& parameter_layout(int64_t param_no) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_2(mht_2_v, 226, "", "./tensorflow/compiler/xla/service/computation_layout.h", "parameter_layout");

    return parameter_layouts_[param_no];
  }
  ShapeLayout* mutable_parameter_layout(int64_t param_no) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_3(mht_3_v, 232, "", "./tensorflow/compiler/xla/service/computation_layout.h", "mutable_parameter_layout");

    return &parameter_layouts_[param_no];
  }

  // Returns the number of parameters in the computation.
  int parameter_count() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_4(mht_4_v, 240, "", "./tensorflow/compiler/xla/service/computation_layout.h", "parameter_count");
 return parameter_layouts_.size(); }

  // Returns the ShapeLayouts of the parameters of the computation.
  const std::vector<ShapeLayout>& parameter_layouts() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_5(mht_5_v, 246, "", "./tensorflow/compiler/xla/service/computation_layout.h", "parameter_layouts");

    return parameter_layouts_;
  }

  // Returns the ShapeLayout of a result of the computation.
  const ShapeLayout& result_layout() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_6(mht_6_v, 254, "", "./tensorflow/compiler/xla/service/computation_layout.h", "result_layout");
 return result_layout_; }
  ShapeLayout* mutable_result_layout() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_7(mht_7_v, 258, "", "./tensorflow/compiler/xla/service/computation_layout.h", "mutable_result_layout");
 return &result_layout_; }

  // Returns the shape of the particular parameter or result of the computation
  // with layout.
  const Shape& parameter_shape(int64_t param_no) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_8(mht_8_v, 265, "", "./tensorflow/compiler/xla/service/computation_layout.h", "parameter_shape");

    return parameter_layouts_[param_no].shape();
  }
  const Shape& result_shape() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_9(mht_9_v, 271, "", "./tensorflow/compiler/xla/service/computation_layout.h", "result_shape");
 return result_layout_.shape(); }

  // Sets layouts of all parameters and the result to the default layout.
  void SetToDefaultLayout();

  void SetToDefaultLayoutIfEmpty();

  // Returns true if all layouts (parameters and result) have been set.
  bool LayoutIsSet() const;

  // Returns a string representation of this object.
  std::string ToString() const;

  // Create a ProgramShape proto based on the parameter and result shapes held
  // within this object.
  ProgramShape ComputeProgramShape() const;

  bool operator==(const ComputationLayout& other) const;
  bool operator!=(const ComputationLayout& other) const;

  template <typename H>
  friend H AbslHashValue(H h, const ComputationLayout& computation_layout) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScomputation_layoutDTh mht_10(mht_10_v, 295, "", "./tensorflow/compiler/xla/service/computation_layout.h", "AbslHashValue");

    h = H::combine(std::move(h), computation_layout.result_layout_.shape());
    for (const auto& parameter_layout : computation_layout.parameter_layouts_) {
      h = H::combine(std::move(h), parameter_layout.shape());
    }
    h = H::combine(std::move(h), computation_layout.parameter_layouts_.size());
    return h;
  }

 private:
  std::vector<ShapeLayout> parameter_layouts_;
  ShapeLayout result_layout_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_LAYOUT_H_
