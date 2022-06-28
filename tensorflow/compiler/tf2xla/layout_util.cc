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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSlayout_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSlayout_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSlayout_utilDTcc() {
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

#include "tensorflow/compiler/tf2xla/layout_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

XlaShapeLayoutHelpers::LayoutPreferenceFn UseNoPreferenceLayoutFn() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSlayout_utilDTcc mht_0(mht_0_v, 193, "", "./tensorflow/compiler/tf2xla/layout_util.cc", "UseNoPreferenceLayoutFn");

  return [](const TensorShape& shape, DataType dtype,
            absl::optional<XlaArgument::Kind>) -> XlaLayoutPreference {
    return XlaLayoutPreference::kNoPreference;
  };
}

// Rewrites the layout of xla_shape if there is tiled sharding.
Status RewriteLayoutWithShardedShape(
    const absl::optional<xla::HloSharding>& sharding, bool use_fast_memory,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    xla::Shape* xla_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSlayout_utilDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/tf2xla/layout_util.cc", "RewriteLayoutWithShardedShape");

  if (sharding && !sharding->IsTileMaximal() && !sharding->IsManual()) {
    // After sharding, per core shape might have different layout. For example,
    // before sharding, a shape [128, 128] will be assigned default
    // minor-to-major {1, 0}. But after we shard this shape to [128, 64] * 2,
    // the sharded shapes will have minor-to-major {0, 1}.
    //
    // As a result, for sharded shapes, we set their layout to per core shape's
    // layout.
    //
    // TODO(endlessroad): for variable input & update, we might have
    // different layouts which will prevent input output aliasing and
    // increase memory usage. Investigate such cases.
    int64_t device = *sharding->tile_assignment().begin();
    std::vector<int64_t> offset =
        sharding->TileOffsetForDevice(*xla_shape, device);
    std::vector<int64_t> limit =
        sharding->TileLimitForDevice(*xla_shape, device);
    std::vector<int64_t> dimensions(xla_shape->rank());
    for (int64_t i = 0; i < xla_shape->rank(); ++i) {
      dimensions[i] = limit[i] - offset[i];
    }
    xla::Shape per_device_xla_shape =
        xla::ShapeUtil::MakeShape(xla_shape->element_type(), dimensions);
    TensorShape per_device_tensor_shape;
    TF_RETURN_IF_ERROR(
        XLAShapeToTensorShape(per_device_xla_shape, &per_device_tensor_shape));
    TF_ASSIGN_OR_RETURN(DataType dtype, EncodePrimitiveTypeAsDataType(
                                            xla_shape->element_type()));
    auto layout_preference = shape_determination_fns.layout_preference_fn(
        per_device_tensor_shape, dtype, absl::nullopt);
    TF_ASSIGN_OR_RETURN(per_device_xla_shape,
                        shape_determination_fns.shape_representation_fn(
                            per_device_tensor_shape, dtype, use_fast_memory,
                            layout_preference));
    *xla_shape->mutable_layout() = per_device_xla_shape.layout();
  }
  return Status::OK();
}

Status RewriteLayoutWithShardedShape(
    const absl::optional<xla::HloSharding>& sharding, bool use_fast_memory,
    XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    xla::Shape* xla_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSlayout_utilDTcc mht_2(mht_2_v, 253, "", "./tensorflow/compiler/tf2xla/layout_util.cc", "RewriteLayoutWithShardedShape");

  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns{
      UseNoPreferenceLayoutFn(), shape_representation_fn};
  return RewriteLayoutWithShardedShape(sharding, use_fast_memory,
                                       shape_determination_fns, xla_shape);
}

// There is a shape_representation_fn or sharding for an output, this function
// uses a reshape to fix the layout.
StatusOr<xla::XlaOp> ReshapeWithCorrectRepresentationAndSharding(
    xla::XlaBuilder* builder, xla::XlaOp original, xla::Shape original_shape,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    absl::optional<xla::OpSharding> sharding, bool fast_mem) {
  if (original_shape.IsTuple()) {
    std::vector<xla::XlaOp> elements;
    for (int i = 0; i < original_shape.tuple_shapes_size(); ++i) {
      auto subsharding = sharding ? sharding->tuple_shardings(i) : sharding;
      TF_ASSIGN_OR_RETURN(auto element,
                          ReshapeWithCorrectRepresentationAndSharding(
                              builder, xla::GetTupleElement(original, i),
                              original_shape.tuple_shapes(i),
                              shape_determination_fns, subsharding, fast_mem));
      elements.push_back(element);
    }
    return xla::Tuple(builder, elements);
  }
  if (!original_shape.IsArray()) return original;
  TensorShape shape;
  TF_RETURN_IF_ERROR(XLAShapeToTensorShape(original_shape, &shape));
  TF_ASSIGN_OR_RETURN(DataType dtype, EncodePrimitiveTypeAsDataType(
                                          original_shape.element_type()));
  auto layout_preference =
      shape_determination_fns.layout_preference_fn(shape, dtype, absl::nullopt);
  TF_ASSIGN_OR_RETURN(auto to_shape,
                      shape_determination_fns.shape_representation_fn(
                          shape, dtype, fast_mem, layout_preference));
  if (sharding) {
    TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                        xla::HloSharding::FromProto(*sharding));

    TF_RETURN_IF_ERROR(RewriteLayoutWithShardedShape(
        hlo_sharding, fast_mem, shape_determination_fns, &to_shape));
  }
  if (xla::ShapeUtil::Compatible(original_shape, to_shape)) {
    for (int64_t i = 0; i < original_shape.rank(); ++i) {
      to_shape.set_dynamic_dimension(i, original_shape.is_dynamic_dimension(i));
    }
  }
  return xla::Reshape(to_shape, original);
}

StatusOr<xla::XlaOp> ReshapeWithCorrectRepresentationAndSharding(
    xla::XlaBuilder* builder, xla::XlaOp original, xla::Shape original_shape,
    XlaHelpers::ShapeRepresentationFn shape_representation_fn,
    absl::optional<xla::OpSharding> sharding, bool fast_mem) {
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns{
      UseNoPreferenceLayoutFn(), shape_representation_fn};
  return ReshapeWithCorrectRepresentationAndSharding(
      builder, original, original_shape, shape_determination_fns, sharding,
      fast_mem);
}

}  // namespace tensorflow
