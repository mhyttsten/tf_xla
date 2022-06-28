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
class MHTracer_DTPStensorflowPScorePStpuPSopsPStpu_partitioned_input_opDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSopsPStpu_partitioned_input_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSopsPStpu_partitioned_input_opDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

ShapeHandle _UpdatePartitionDim(InferenceContext* c, const ShapeHandle handle,
                                const int partition_dim) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSopsPStpu_partitioned_input_opDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/tpu/ops/tpu_partitioned_input_op.cc", "_UpdatePartitionDim");

  ShapeHandle newoutput0;
  shape_inference::DimensionHandle new_dim;
  TF_CHECK_OK(
      c->Multiply(c->Dim(handle, partition_dim), c->num_inputs(), &new_dim));
  TF_CHECK_OK(c->ReplaceDim(handle, partition_dim, new_dim, &newoutput0));
  return newoutput0;
}

REGISTER_OP("TPUPartitionedInput")
    .Input("inputs: N * T")
    .Output("output: T")
    .Attr("N: int >= 1")
    .Attr("T: type")
    .Attr("partition_dim: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));
      int partition_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("partition_dim", &partition_dim));

      ShapeHandle cur = c->input(c->num_inputs() - 1);
      for (int i = c->num_inputs() - 2; i >= 0; --i) {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Merge(c->input(i), cur, &cur),
                                        "From merging shape ", i,
                                        " with other shapes.");
      }
      if (partition_dim == -1 || dtype == DT_RESOURCE) {
        c->set_output(0, cur);
      } else {
        ShapeHandle newoutput0 = _UpdatePartitionDim(c, cur, partition_dim);
        c->set_output(0, newoutput0);
      }

      // If this is a resource, unify the resource shapes.
      if (dtype == DT_RESOURCE) {
        ShapeHandle previous_shape_handle;
        const std::vector<shape_inference::ShapeAndType>* shapes_and_types =
            nullptr;
        for (int i = c->num_inputs() - 1; i >= 0; --i) {
          shapes_and_types = c->input_handle_shapes_and_types(i);
          if (shapes_and_types) {
            ShapeHandle shape_handle = shapes_and_types->at(0).shape;
            if (!c->FullyDefined(shape_handle)) {
              return errors::InvalidArgument("Inputs must have static shape,",
                                             "input[", i,
                                             "] has unknown dimension.");
            }
            if (i != c->num_inputs() - 1) {
              ShapeHandle tmp;
              if (!c->Merge(shape_handle, previous_shape_handle, &tmp).ok()) {
                return errors::InvalidArgument(
                    "Inputs must have the same shape.");
              }
            } else {
              previous_shape_handle = shape_handle;
            }
          }
        }
        if (shapes_and_types) {
          if (partition_dim == -1) {
            c->set_output_handle_shapes_and_types(0, *shapes_and_types);
          } else {
            ShapeHandle newoutput0 =
                _UpdatePartitionDim(c, previous_shape_handle, partition_dim);

            std::vector<shape_inference::ShapeAndType> output_shapes_and_types;
            output_shapes_and_types.push_back(shape_inference::ShapeAndType(
                newoutput0, shapes_and_types->at(0).dtype));
            c->set_output_handle_shapes_and_types(0, output_shapes_and_types);
          }
        }
      }

      return Status::OK();
    });

}  // namespace tensorflow
