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

#ifndef TENSORFLOW_COMPILER_XLA_SHAPE_LAYOUT_H_
#define TENSORFLOW_COMPILER_XLA_SHAPE_LAYOUT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSshape_layoutDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_layoutDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSshape_layoutDTh() {
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

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

// A ShapeLayout object encapsulates the layout of a particular shape (including
// tuples). This differs from the Layout proto which describes the layout of a
// single array. ShapeLayout contains a Layout proto for each array in the shape
// (a tuple can have more than one array). For array shapes, this object
// trivially holds a single Layout. Logically, ShapeLayout holds a nonmutable
// shape with mutable layouts.
class ShapeLayout {
 public:
  // Constructs a ShapeLayout of the given shape. Layouts are copied from the
  // shape parameter.
  explicit ShapeLayout(const Shape& shape) : shape_(shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_layoutDTh mht_0(mht_0_v, 207, "", "./tensorflow/compiler/xla/shape_layout.h", "ShapeLayout");
}

  // Assigns the layouts in this ShapeLayout to the Layout fields of the given
  // shape. 'to_shape' and the shape of the ShapeLayout object must be
  // compatible.
  Status AssignLayoutToShape(Shape* to_shape) const;

  // Returns true if the Layouts in this ShapeLayout match the layouts in the
  // given shape. Returns false otherwise. If the given shape is not compatible
  // with the ShapeLayout's shape, then false is returned. If
  // `ignore_fully_empty_tiling` is true, tiling info is ignored if one of the
  // shapes has no tiling at all in all its subshapes.
  bool MatchesLayoutInShape(const Shape& shape,
                            bool minor_to_major_only = false,
                            bool ignore_fully_empty_tiling = false) const;

  // Copies the layout from the given shape into this ShapeLayout. 'other_shape'
  // must be compatible with the ShapeLayout's shape.
  Status CopyLayoutFromShape(const Shape& other_shape);

  // Clears (Layout::Clear) all the Layouts stored in this object.
  void Clear();

  // Sets all Layouts stored in this object to the default layout.
  void SetToDefaultLayout();

  // Returns the shape (with layouts).
  const Shape& shape() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_layoutDTh mht_1(mht_1_v, 237, "", "./tensorflow/compiler/xla/shape_layout.h", "shape");
 return shape_; }

  // Clear dynamic dimensions of this module. Pretending the module creates
  // static results. Useful in inspecting full outputs when testing.
  void ClearDynamicShape() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_layoutDTh mht_2(mht_2_v, 244, "", "./tensorflow/compiler/xla/shape_layout.h", "ClearDynamicShape");
 shape_.clear_dynamic_dimensions(); }

  // Checks that a layout is set for the shape, and returns a reference to the
  // layout directly on the shape. Shape must not be a tuple.
  const Layout& layout() const;

  // Returns true if all layouts have been set for this ShapeLayout object. That
  // is, every array has a layout.
  bool LayoutIsSet() const;

  // Resets the layout on the shape to the provided layout. Shape must not be a
  // tuple.
  void ResetLayout(const Layout& layout);

  // Resets the layout on the shape at the provided ShapeIndex to the provided
  // layout. Shape must be a tuple.
  void ResetLayout(const Layout& layout, ShapeIndexView shape_index);

  // Returns a string representation of this object.
  std::string ToString() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSshape_layoutDTh mht_3(mht_3_v, 266, "", "./tensorflow/compiler/xla/shape_layout.h", "ToString");

    return ShapeUtil::HumanStringWithLayout(shape_);
  }

  // Tests for equality of both shape and layout (ShapeUtil::Equal).
  bool operator==(const ShapeLayout& other) const;
  bool operator!=(const ShapeLayout& other) const;

 private:
  Shape shape_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_LAYOUT_H_
