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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc() {
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

#include "tensorflow/compiler/tf2xla/shape_util.h"

#include <numeric>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {

Status PopulateInfeedLayoutVector(const xla::Shape& shape,
                                  std::vector<int>* layouts) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/compiler/tf2xla/shape_util.cc", "PopulateInfeedLayoutVector");

  if (shape.IsTuple()) {
    int64_t tuple_elements = xla::ShapeUtil::TupleElementCount(shape);
    for (int64_t i = 0; i < tuple_elements; ++i) {
      const xla::Shape& subshape =
          xla::ShapeUtil::GetTupleElementShape(shape, i);
      TF_RETURN_IF_ERROR(PopulateInfeedLayoutVector(subshape, layouts));
    }
  } else if (xla::LayoutUtil::HasLayout(shape)) {
    for (auto dim : xla::LayoutUtil::MinorToMajor(shape)) {
      layouts->push_back(dim);
    }
  } else {
    layouts->insert(layouts->end(), shape.rank(), -1);
  }
  return Status::OK();
}

// Populate the output layout unless the minor_to_major array contains all -1
// value, in which case the layout is considered missing and the API returns
// false.
StatusOr<bool> MakeLayout(absl::Span<const int64_t> minor_to_major,
                          xla::Layout* layout) {
  if (std::all_of(minor_to_major.begin(), minor_to_major.end(),
                  [](int64_t dim) { return dim == -1; })) {
    return false;
  }
  std::vector<bool> dim_present(minor_to_major.size(), false);
  for (auto dim : minor_to_major) {
    const int minor_to_major_size = minor_to_major.size();
    if (dim < 0 || dim >= minor_to_major_size) {
      return errors::InvalidArgument("Layout dimension out of range: dim=", dim,
                                     " rank=", minor_to_major.size());
    }
    if (dim_present[dim]) {
      return errors::InvalidArgument("Repeated layout dimension: dim=", dim);
    }
    dim_present[dim] = true;
  }
  *layout = xla::LayoutUtil::MakeLayout(minor_to_major);
  return true;
}

Status AssignLayout(
    absl::Span<const int64_t> minor_to_major,
    const std::function<xla::Layout(const xla::Shape&)>& layout_func,
    xla::Shape* shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc mht_1(mht_1_v, 247, "", "./tensorflow/compiler/tf2xla/shape_util.cc", "AssignLayout");

  xla::Layout layout;
  TF_ASSIGN_OR_RETURN(bool has_layout, MakeLayout(minor_to_major, &layout));
  if (!has_layout && layout_func) {
    layout = layout_func(*shape);
  }
  *shape->mutable_layout() = layout;
  return Status::OK();
}

}  // namespace

// Convert an XLA Shape into the equivalent TensorFlow shape.
Status XLAShapeToTensorShape(const xla::Shape& shape,
                             TensorShape* tensor_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/tf2xla/shape_util.cc", "XLAShapeToTensorShape");

  if (shape.IsTuple()) {
    return errors::InvalidArgument("XLA shape ",
                                   xla::ShapeUtil::HumanString(shape),
                                   " cannot be converted to a TensorShape");
  }
  *tensor_shape = TensorShape();
  for (int i = 0; i < shape.rank(); ++i) {
    tensor_shape->AddDim(shape.dimensions(i));
  }
  return Status::OK();
}

// Convert a TensorShape into the equivalent XLA Shape proto.
Status TensorShapeToXLAShape(DataType dtype,
                             const PartialTensorShape& tensor_shape,
                             xla::Shape* shape) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc mht_3(mht_3_v, 283, "", "./tensorflow/compiler/tf2xla/shape_util.cc", "TensorShapeToXLAShape");

  xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));
  *shape = TensorShapeToXLAShape(type, tensor_shape);
  return Status::OK();
}

xla::Shape TensorShapeToXLAShape(xla::PrimitiveType type,
                                 const PartialTensorShape& tensor_shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc mht_4(mht_4_v, 294, "", "./tensorflow/compiler/tf2xla/shape_util.cc", "TensorShapeToXLAShape");

  if (tensor_shape.unknown_rank()) {
    // For unknown shape, create a rank 1 size 0 tensor.
    return xla::ShapeUtil::MakeShapeWithLayout(type, {0}, {0});
  }
  int rank = tensor_shape.dims();
  std::vector<int64_t> dimensions(rank);
  std::vector<bool> dynamic_dimensions(rank, false);
  std::vector<int64_t> layout(rank);
  for (int d = 0; d < rank; ++d) {
    dimensions[d] = tensor_shape.dim_size(d);
    if (dimensions[d] < 0) {
      dynamic_dimensions[d] = true;
      // TODO(b/177329258): Consider improving this/enabling MakeShapeWithLayout
      // to work wuith dynamic shapes.
      LOG(WARNING) << "Unable to convert TF shape with dynamic size to XLA "
                      "shape; returning unknown sentinel value";
      return xla::ShapeUtil::MakeShapeWithLayout(type, {0}, {0});
    }
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);
  xla::Shape result =
      xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);

  for (int64_t d = 0; d < rank; ++d) {
    result.set_dynamic_dimension(d, dynamic_dimensions[d]);
  }
  return result;
}

// Convert a TensorShape into the equivalent XLA Shape proto.
Status TensorShapeToXLAShape(DataType dtype, const TensorShape& tensor_shape,
                             xla::Shape* shape) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc mht_5(mht_5_v, 330, "", "./tensorflow/compiler/tf2xla/shape_util.cc", "TensorShapeToXLAShape");

  xla::PrimitiveType type;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(dtype, &type));
  *shape = TensorShapeToXLAShape(type, tensor_shape);
  return Status::OK();
}

xla::Shape TensorShapeToXLAShape(xla::PrimitiveType type,
                                 const TensorShape& tensor_shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc mht_6(mht_6_v, 341, "", "./tensorflow/compiler/tf2xla/shape_util.cc", "TensorShapeToXLAShape");

  int rank = tensor_shape.dims();
  std::vector<int64_t> dimensions(rank);
  std::vector<int64_t> layout(rank);
  for (int d = 0; d < rank; ++d) {
    dimensions[d] = tensor_shape.dim_size(d);
  }
  // XLA uses minor-to-major; Tensorflow uses major-to-minor.
  std::iota(layout.rbegin(), layout.rend(), 0);

  return xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, layout);
}

StatusOr<std::vector<int>> GetShapeLayoutVector(const xla::Shape& shape) {
  std::vector<int> layouts;
  TF_RETURN_IF_ERROR(PopulateInfeedLayoutVector(shape, &layouts));
  return layouts;
}

Status GetShapeWithLayout(
    const xla::Shape& input_shape, absl::Span<const int64_t> minor_to_major,
    const std::function<xla::Layout(const xla::Shape&)>& layout_func,
    xla::Shape* output_shape) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSshape_utilDTcc mht_7(mht_7_v, 366, "", "./tensorflow/compiler/tf2xla/shape_util.cc", "GetShapeWithLayout");

  if (input_shape.IsTuple()) {
    int64_t tuple_elements = xla::ShapeUtil::TupleElementCount(input_shape);
    std::vector<xla::Shape> shapes;
    shapes.reserve(tuple_elements);
    size_t position = 0;
    for (int64_t i = 0; i < tuple_elements; ++i) {
      const xla::Shape& shape =
          xla::ShapeUtil::GetTupleElementShape(input_shape, i);
      if (shape.IsTuple()) {
        return errors::InvalidArgument(
            "Nested tuples not supported: ",
            xla::ShapeUtil::HumanString(input_shape));
      }
      int64_t rank = shape.rank();
      if (position + rank > minor_to_major.size()) {
        return errors::InvalidArgument(
            "Not enough layout attribute elements: position=", position,
            " rank=", rank, " elements=", minor_to_major.size());
      }
      shapes.push_back(shape);
      TF_RETURN_IF_ERROR(AssignLayout(
          absl::Span<const int64_t>(minor_to_major).subspan(position, rank),
          layout_func, &shapes.back()));
      position += rank;

      VLOG(4) << "Shape[" << i
              << "] = " << xla::ShapeUtil::HumanStringWithLayout(shapes.back());
    }
    if (position != minor_to_major.size()) {
      return errors::InvalidArgument(
          "Too many elements passed in the layout attribute: position=",
          position, " size=", minor_to_major.size());
    }
    *output_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  } else {
    int64_t rank = input_shape.rank();
    const int64_t minor_to_major_size = minor_to_major.size();
    if (rank != minor_to_major_size) {
      return errors::InvalidArgument(
          "Wrong number of layout attribute elements: rank=", rank,
          " elements=", minor_to_major.size());
    }
    *output_shape = input_shape;
    TF_RETURN_IF_ERROR(AssignLayout(minor_to_major, layout_func, output_shape));

    VLOG(4) << "Shape[] = "
            << xla::ShapeUtil::HumanStringWithLayout(*output_shape);
  }
  return Status::OK();
}

}  // namespace tensorflow
