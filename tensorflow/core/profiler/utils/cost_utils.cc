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
class MHTracer_DTPStensorflowPScorePSprofilerPSutilsPScost_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPScost_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSutilsPScost_utilsDTcc() {
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

#include "tensorflow/core/profiler/utils/cost_utils.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/costs/cost_estimator.h"
#include "tensorflow/core/grappler/costs/op_context.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

namespace {

// Decode the string that encodes tensor shape and type information and convert
// to TensorProperties.
// Returns an empty TensorProperties if error or input is "".
// See OpKernel::TraceString() to see when the shape is encoded as "".
// Input format is <DTYPE>[<dim1>, <dim2>,...]
static OpInfo::TensorProperties GetTensorProperties(absl::string_view info) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("info: \"" + std::string(info.data(), info.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPScost_utilsDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/profiler/utils/cost_utils.cc", "GetTensorProperties");

  OpInfo::TensorProperties tensor_prop;
  std::vector<absl::string_view> parts = absl::StrSplit(info, '[');
  if (parts.size() != 2) return tensor_prop;
  DataType data_type = DT_INVALID;
  if (!DataTypeFromString(parts[0], &data_type)) return tensor_prop;
  tensor_prop.set_dtype(data_type);
  absl::ConsumeSuffix(&parts[1], "]");
  if (parts[1].empty()) {  // Scalar type.
    tensor_prop.mutable_shape()->add_dim()->set_size(1);
    return tensor_prop;
  }
  std::vector<absl::string_view> dims = absl::StrSplit(parts[1], ',');
  for (const auto dim : dims) {
    int size;
    if (!absl::SimpleAtoi(dim, &size)) return OpInfo::TensorProperties();
    tensor_prop.mutable_shape()->add_dim()->set_size(size);
  }
  return tensor_prop;
}

}  // namespace

TfOpRoofLineCostEstimator::~TfOpRoofLineCostEstimator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPScost_utilsDTcc mht_1(mht_1_v, 246, "", "./tensorflow/core/profiler/utils/cost_utils.cc", "TfOpRoofLineCostEstimator::~TfOpRoofLineCostEstimator");

  if (!unsupported_ops_.empty()) {
    LOG(ERROR) << "Unsupported Op for Roofline Cost Analysis are:"
               << absl::StrJoin(unsupported_ops_, ",");
  }
}

grappler::DeviceInfo TfOpRoofLineCostEstimator::GetDeviceInfo(
    const DeviceProperties& device) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPScost_utilsDTcc mht_2(mht_2_v, 257, "", "./tensorflow/core/profiler/utils/cost_utils.cc", "TfOpRoofLineCostEstimator::GetDeviceInfo");

  // Hypothetical devices that is used to measure peak flops and memory bytes
  // accessed.
  return grappler::DeviceInfo(/*gigaops=*/1, /*gb_per_sec=*/1);
}

TfOpRoofLineCostEstimator::OpRoofLineStats TfOpRoofLineCostEstimator::Predict(
    const XEventVisitor& event) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSutilsPScost_utilsDTcc mht_3(mht_3_v, 267, "", "./tensorflow/core/profiler/utils/cost_utils.cc", "TfOpRoofLineCostEstimator::Predict");

  TfOp tf_op;
  absl::string_view tensor_shapes;
  event.ForEachStat([&](const XStatVisitor& stat) {
    if (!stat.Type().has_value()) return;
    switch (stat.Type().value()) {
      case StatType::kTfOp:
        tf_op = ParseTfOpFullname(stat.StrOrRefValue());
        break;
      case StatType::kTensorShapes:
        tensor_shapes = stat.StrOrRefValue();
        break;
    }
  });

  // Return empty OpRoofLineStats if shape is not traced or this is not a tf op.
  if (tf_op.type.empty() || tensor_shapes.empty()) {
    return {0ULL, 0ULL, /*inaccurate=*/true};
  }

  grappler::OpContext op_context;
  op_context.name = std::string(tf_op.type);
  op_context.op_info.set_op(op_context.name);
  for (absl::string_view tensor : ParseTensorShapes(tensor_shapes)) {
    *op_context.op_info.add_inputs() = GetTensorProperties(tensor);
  }
  grappler::Costs costs = PredictCosts(op_context);
  if (costs.inaccurate) unsupported_ops_.insert(std::string(tf_op.type));

  VLOG(1) << tf_op.type << tensor_shapes
          << " flops:" << costs.compute_time.count()
          << " bytes:" << costs.memory_time.count();

  /* The compute_time is measured in nanoseconds, therefore numerically it is
   * equal to flops because giga ops / second cancel the nanoseconds.
   * Same for memory_time */
  return {/*flops=*/static_cast<uint64>(costs.compute_time.count()),
          /*bytes_accessed=*/static_cast<uint64>(costs.memory_time.count()),
          /*inaccurate=*/costs.inaccurate};
}

}  // namespace profiler
}  // namespace tensorflow
