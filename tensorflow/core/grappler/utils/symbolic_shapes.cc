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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc() {
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

#include "tensorflow/core/grappler/utils/symbolic_shapes.h"

#include <unordered_map>

#include "tensorflow/core/util/bcast.h"

namespace tensorflow {
namespace grappler {
namespace {

BCast::Vec ShapeDims(const TensorShapeProto& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "ShapeDims");

  BCast::Vec dims;
  dims.reserve(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i)
    dims.push_back(shape.dim(i).size());
  return dims;
}

}  // namespace

bool IsKnown(const TensorShapeProto::Dim& dim) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_1(mht_1_v, 208, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "IsKnown");
 return dim.size() >= 0; }

bool IsKnownSymbolically(const TensorShapeProto::Dim& dim) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "IsKnownSymbolically");

  return dim.size() <= -2;
}

bool IsUnknown(const TensorShapeProto::Dim& dim) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_3(mht_3_v, 220, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "IsUnknown");
 return dim.size() == -1; }

bool ShapeIsSymbolicallyDefined(const TensorShapeProto& shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_4(mht_4_v, 225, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "ShapeIsSymbolicallyDefined");

  return !shape.unknown_rank() &&
         std::all_of(
             shape.dim().begin(), shape.dim().end(),
             [](const TensorShapeProto::Dim& dim) { return !IsUnknown(dim); });
}

bool ShapeIsSymbolicallyDefined(const OpInfo::TensorProperties& properties) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_5(mht_5_v, 235, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "ShapeIsSymbolicallyDefined");

  return ShapeIsSymbolicallyDefined(properties.shape());
}

int Rank(const TensorShapeProto& shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_6(mht_6_v, 242, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "Rank");

  if (shape.unknown_rank()) {
    return -1;
  }
  return shape.dim_size();
}

int64_t NumCoefficients(const TensorShapeProto& shape) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_7(mht_7_v, 252, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "NumCoefficients");

  if (shape.unknown_rank()) {
    return -1;
  }
  int64_t num_coefficients = 1;
  for (const auto& dim : shape.dim()) {
    if (dim.size() < 0) {
      return -1;
    }
    num_coefficients *= dim.size();
  }
  return num_coefficients;
}

bool ShapesSymbolicallyEqual(const TensorShapeProto& left,
                             const TensorShapeProto& right) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_8(mht_8_v, 270, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "ShapesSymbolicallyEqual");

  if (left.unknown_rank() || right.unknown_rank() ||
      left.dim_size() != right.dim_size()) {
    return false;
  }
  for (int i = 0; i < left.dim_size(); ++i) {
    const auto& ldim = left.dim(i);
    const auto& rdim = right.dim(i);
    if (IsUnknown(ldim) || IsUnknown(rdim) || ldim.size() != rdim.size()) {
      return false;
    }
  }
  return true;
}

bool ShapesSymbolicallyEqual(const OpInfo::TensorProperties& left,
                             const OpInfo::TensorProperties& right) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_9(mht_9_v, 289, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "ShapesSymbolicallyEqual");

  return ShapesSymbolicallyEqual(left.shape(), right.shape());
}

bool ShapesBroadcastable(const TensorShapeProto& left,
                         const TensorShapeProto& right) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_10(mht_10_v, 297, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "ShapesBroadcastable");

  if (!ShapeIsSymbolicallyDefined(left) || !ShapeIsSymbolicallyDefined(right)) {
    return false;
  }
  BCast bcast(ShapeDims(left), ShapeDims(right),
              /*fewer_dims_optimization*/ false);
  return bcast.IsValid();
}

bool ShapesBroadcastable(const OpInfo::TensorProperties& left,
                         const OpInfo::TensorProperties& right) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_11(mht_11_v, 310, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "ShapesBroadcastable");

  return ShapesBroadcastable(left.shape(), right.shape());
}

bool ShapeAfterBroadcast(const TensorShapeProto& left,
                         const TensorShapeProto& right,
                         TensorShapeProto* output_shape) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_12(mht_12_v, 319, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "ShapeAfterBroadcast");

  if (!ShapeIsSymbolicallyDefined(left) || !ShapeIsSymbolicallyDefined(right)) {
    return false;
  }
  BCast bcast(ShapeDims(left), ShapeDims(right),
              /*fewer_dims_optimization*/ false);
  if (!bcast.IsValid()) {
    return false;
  }
  output_shape->set_unknown_rank(false);
  output_shape->clear_dim();
  for (const auto& dim : bcast.output_shape()) {
    output_shape->add_dim()->set_size(dim);
  }
  return true;
}

bool CompareSymbolicallyShapedTensorSizes(const TensorShapeProto& left,
                                          const TensorShapeProto& right) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_13(mht_13_v, 340, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "CompareSymbolicallyShapedTensorSizes");

  // if one of the ranks is unknown, it's impossible to compare tensor sizes
  if (left.unknown_rank() || right.unknown_rank()) {
    return false;
  }

  // Tensor size, computed as a product of defined dimensions
  int64_t left_defined_size = 1;
  int64_t right_defined_size = 1;

  // Keep how many times each unknown dimension appeared on the left and right
  std::unordered_map<int64_t, int64_t> left_unknown_dims;
  std::unordered_map<int64_t, int64_t> right_unknown_dims;

  // Assign unique id to every unknown dimension (-1). We are going to
  // assign positive ids, because negative values are already used by
  // symbolic dimensions.
  int64_t unknown_dim_id = 1;

  // For each shape dimension update "defined tensor size", if shape is defined,
  // or increment a counter for unknown dim.
  auto process_dimensions =
      [&unknown_dim_id](const TensorShapeProto& shape, int64* defined_size,
                        std::unordered_map<int64, int64>* unknown_dims) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_14(mht_14_v, 366, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "lambda");

        for (int i = 0; i < shape.dim_size(); ++i) {
          const auto& dim = shape.dim(i);
          int64_t dim_size = dim.size();
          if (dim_size > 0) {
            *defined_size *= dim_size;
          } else if (IsUnknown(dim)) {
            ++(*unknown_dims)[unknown_dim_id++];
          } else if (IsKnownSymbolically(dim)) {
            ++(*unknown_dims)[dim_size];
          }
        }
      };

  process_dimensions(left, &left_defined_size, &left_unknown_dims);
  process_dimensions(right, &right_defined_size, &right_unknown_dims);

  // Compute a union of unknown dimension ids appeared in both shapes
  std::set<int64_t> unknown_dims;
  for (const auto& el : left_unknown_dims) unknown_dims.insert(el.first);
  for (const auto& el : right_unknown_dims) unknown_dims.insert(el.first);

  // Cancel unknown dimensions that appeared in both shapes
  for (int64_t unknown_dim : unknown_dims) {
    int64_t co_occurrence = std::min(left_unknown_dims[unknown_dim],
                                     right_unknown_dims[unknown_dim]);
    left_unknown_dims[unknown_dim] -= co_occurrence;
    right_unknown_dims[unknown_dim] -= co_occurrence;
  }

  // Count unbalanced unknown dimensions
  int64_t left_unbalanced_unknown_dims = 0;
  int64_t right_unbalanced_unknown_dims = 0;
  for (const auto& el : left_unknown_dims)
    left_unbalanced_unknown_dims += el.second;
  for (const auto& el : right_unknown_dims)
    right_unbalanced_unknown_dims += el.second;

  if (left_unbalanced_unknown_dims == 0 && right_unbalanced_unknown_dims == 0) {
    // If unknown dimensions cancelled each other, compare tensor sizes
    // represented by defined dimensions
    return left_defined_size < right_defined_size;
  }

  if (left_defined_size <= right_defined_size &&
      left_unbalanced_unknown_dims == 0 && right_unbalanced_unknown_dims > 0) {
    // If size of a 'left" tensor computed from defined dimensions less or
    // equal, and shape on the right has unbalanced unknown dimensions, we can
    // guarantee that shape on the left is strictly smaller (assuming that
    // unknown dimension size is larger than 1)
    return true;
  }

  // In every other case, assuming that unknown dimensions can be arbitrary
  // large in size, we can't guarantee any ordering
  return false;
}

bool CompareSymbolicallyShapedTensorSizes(
    const OpInfo::TensorProperties& left,
    const OpInfo::TensorProperties& right) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_15(mht_15_v, 429, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "CompareSymbolicallyShapedTensorSizes");

  return CompareSymbolicallyShapedTensorSizes(left.shape(), right.shape());
}

int64_t ComputeSizeRatio(const TensorShapeProto& numerator,
                         const TensorShapeProto& denominator) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSsymbolic_shapesDTcc mht_16(mht_16_v, 437, "", "./tensorflow/core/grappler/utils/symbolic_shapes.cc", "ComputeSizeRatio");

  if (numerator.unknown_rank() || denominator.unknown_rank()) {
    return -1;
  }
  std::multiset<int> symbolic_dims;
  int64_t num = 1;
  for (const auto& dim : numerator.dim()) {
    if (dim.size() == -1) {
      return -1;
    } else if (dim.size() < -1) {
      symbolic_dims.insert(dim.size());
    } else {
      num *= dim.size();
    }
  }
  int64_t denom = 1;
  for (const auto& dim : denominator.dim()) {
    if (dim.size() == -1) {
      return -1;
    } else if (dim.size() < -1) {
      auto it = symbolic_dims.find(dim.size());
      if (it == symbolic_dims.end()) {
        return -1;
      }
      symbolic_dims.erase(it);
    } else {
      denom *= dim.size();
    }
  }
  if (denom == 0) {
    return -1;
  }
  if (!symbolic_dims.empty()) {
    return -1;
  }
  return num / denom;
}

}  // end namespace grappler
}  // end namespace tensorflow
