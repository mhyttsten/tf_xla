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
class MHTracer_DTPStensorflowPScorePSopsPSrandom_index_shuffle_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSrandom_index_shuffle_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSrandom_index_shuffle_opsDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

static Status StatelessRandomPermuteShape(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSrandom_index_shuffle_opsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/ops/random_index_shuffle_ops.cc", "StatelessRandomPermuteShape");

  ShapeHandle index_shape, seed_shape, max_index_shape;

  // Basic constraints but unknown ranks will not raise errors here.
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &index_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &seed_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 2, &seed_shape));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &max_index_shape));

  // Figure out if the output is a scalar or tensor.
  const int32 index_rank = c->Rank(index_shape);
  const int32 seed_rank = c->Rank(seed_shape);
  const int32 max_index_rank = c->Rank(max_index_shape);

  // Check that last dimension of seed is 3.
  if (seed_rank == 1 && c->Value(c->Dim(seed_shape, 0)) != 3) {
    return errors::InvalidArgument("Seed must have shape [3] but got [",
                                   c->Value(c->Dim(seed_shape, 0)), "].");
  }
  if (seed_rank == 2 && c->Value(c->Dim(seed_shape, 1)) != 3) {
    return errors::InvalidArgument("Seed must have shape [n, 3] but got [",
                                   c->Value(c->Dim(seed_shape, 0)), ", ",
                                   c->Value(c->Dim(seed_shape, 1)), "].");
  }

  // If all inputs are scalars the output is a scalar.
  const bool output_is_scalar =
      (index_rank == 0 && seed_rank == 1 && max_index_rank == 0);
  if (output_is_scalar) {
    c->set_output(0, c->Scalar());
    return Status::OK();
  }

  if (!c->FullyDefined(index_shape) || !c->FullyDefined(seed_shape) ||
      !c->FullyDefined(max_index_shape)) {
    const bool output_is_vector =
        (index_rank == 1 || seed_rank == 2 || max_index_rank == 1);
    if (output_is_vector) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
    }
    return Status::OK();
  }

  // Shape is fully defined and the output is a vector.
  const int64_t num_indices = index_rank ? c->Value(c->Dim(index_shape, 0)) : 1;
  const int64_t num_seeds =
      seed_rank == 2 ? c->Value(c->Dim(seed_shape, 0)) : 1;
  const int64_t num_max_indices =
      max_index_rank ? c->Value(c->Dim(max_index_shape, 0)) : 1;
  const int64_t num_outputs =
      std::max(std::max(num_indices, num_seeds), num_max_indices);
  if (num_indices != 1 && num_indices != num_outputs) {
    return errors::InvalidArgument("Index has shape [", num_indices,
                                   "] but must have shape [", num_outputs,
                                   "].");
  }
  if (num_seeds != 1 && num_seeds != num_outputs) {
    return errors::InvalidArgument("Seed has shape [", num_seeds,
                                   "3, ] but must have shape [", num_outputs,
                                   ", 3].");
  }
  if (num_max_indices != 1 && num_max_indices != num_outputs) {
    return errors::InvalidArgument("Max index has shape [", num_max_indices,
                                   "] but must have shape [", num_outputs,
                                   "].");
  }
  c->set_output(0, c->Vector(num_outputs));
  return Status::OK();
}

REGISTER_OP("RandomIndexShuffle")
    .Input("index: dtype")
    .Input("seed: Tseed")
    .Input("max_index: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int32, uint32, int64, uint64}")
    .Attr("Tseed: {int32, uint32, int64, uint64}")
    .SetShapeFn(StatelessRandomPermuteShape);

}  // namespace
}  // namespace tensorflow
