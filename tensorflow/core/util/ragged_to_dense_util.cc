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
class MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_utilDTcc() {
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

#include "tensorflow/core/util/ragged_to_dense_util.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace tensorflow {

using errors::InvalidArgument;

tensorflow::Status GetRowPartitionTypesHelper(
    const std::vector<string>& row_partition_type_strings,
    std::vector<RowPartitionType>* row_partition_types) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/core/util/ragged_to_dense_util.cc", "GetRowPartitionTypesHelper");

  *row_partition_types = GetRowPartitionTypesHelper(row_partition_type_strings);
  if (row_partition_types->size() != row_partition_type_strings.size()) {
    // Something was not converted, return error status.
    return InvalidArgument(
        "Unknown string for partition info type: ",
        row_partition_type_strings.at(row_partition_types->size()));
  }
  return tensorflow::Status::OK();
}

tensorflow::Status CombineRaggedTensorToTensorShapes(
    int ragged_rank, const TensorShapeProto& shape,
    const TensorShapeProto& value_shape, TensorShapeProto* output_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_utilDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/util/ragged_to_dense_util.cc", "CombineRaggedTensorToTensorShapes");

  // Test for consistency of value_shape and shape specified.
  // If shape is unspecified and value_shape is specified, then copy
  // over the size from the value_shape dimension.

  if (value_shape.unknown_rank() && shape.unknown_rank()) {
    output_shape->Clear();
    output_shape->set_unknown_rank(true);
    return tensorflow::Status::OK();
  }

  if (shape.unknown_rank()) {
    // Here, value_shape must be of known size.
    while (output_shape->dim_size() < ragged_rank + value_shape.dim_size()) {
      output_shape->add_dim()->set_size(-1);
    }
  } else {
    *output_shape = shape;
  }
  if (value_shape.unknown_rank()) {
    return tensorflow::Status::OK();
  }
  // At this point, value_shape and output_shape have known ranks.
  if (ragged_rank + value_shape.dim_size() != output_shape->dim_size()) {
    return InvalidArgument(
        "rt_input.shape and shape=", TensorShape::DebugString(shape),
        " are incompatible: rt_input.rank = ",
        ragged_rank + value_shape.dim_size(),
        " but shape.rank = ", output_shape->dim_size());
  }

  for (int i = 1; i < value_shape.dim_size(); ++i) {
    const TensorShapeProto::Dim& value_dim = value_shape.dim(i);
    TensorShapeProto::Dim* output_shape_dim = output_shape->mutable_dim(
        output_shape->dim_size() - value_shape.dim_size() + i);

    if (value_dim.size() >= 0) {
      if (output_shape_dim->size() >= 0) {
        if (output_shape_dim->size() != value_dim.size()) {
          return InvalidArgument(
              "rt_input.shape and shape=", TensorShape::DebugString(shape),
              " are incompatible: rt_input.shape[", i + ragged_rank,
              "] = ", value_dim.size(), " but shape[", i + ragged_rank,
              "] = ", output_shape_dim->size());
        }
      } else {
        output_shape_dim->set_size(value_dim.size());
      }
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status ValidateDefaultValueShape(
    const TensorShapeProto& default_value_shape,
    const TensorShapeProto& value_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSragged_to_dense_utilDTcc mht_2(mht_2_v, 272, "", "./tensorflow/core/util/ragged_to_dense_util.cc", "ValidateDefaultValueShape");

  if (default_value_shape.unknown_rank() || value_shape.unknown_rank()) {
    return tensorflow::Status::OK();
  }

  int default_ndims = default_value_shape.dim_size();
  int values_ndims = value_shape.dim_size();
  if (default_ndims >= values_ndims) {
    return InvalidArgument(
        "default_value.shape=", TensorShape::DebugString(default_value_shape),
        " and rt_input.flat_values.shape=",
        TensorShape::DebugString(value_shape),
        " are incompatible: default_value.rank = ", default_ndims,
        "  must be less than rt_input.flat_values.rank = ", values_ndims);
  }
  for (int i = 0; i < std::min(default_ndims, values_ndims - 1); ++i) {
    int default_dim = default_value_shape.dim(i).size();
    int value_dim = value_shape.dim(i + 1).size();
    if (default_dim >= 0 && value_dim >= 0 && default_dim != 1 &&
        default_dim != value_dim) {
      return InvalidArgument(
          "default_value.shape=", TensorShape::DebugString(default_value_shape),
          " and rt_input.flat_values.shape=",
          TensorShape::DebugString(value_shape),
          " are incompatible: default_value.shape[",
          i - default_value_shape.dim_size(), "] = ", default_dim,
          " but rt_input.flat_values.shape[",
          i - default_value_shape.dim_size(), "] = ", value_dim);
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace tensorflow
