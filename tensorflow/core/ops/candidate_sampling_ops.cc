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
class MHTracer_DTPStensorflowPScorePSopsPScandidate_sampling_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPScandidate_sampling_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPScandidate_sampling_opsDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

namespace {

Status CandidateSamplerShapeFn(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPScandidate_sampling_opsDTcc mht_0(mht_0_v, 196, "", "./tensorflow/core/ops/candidate_sampling_ops.cc", "CandidateSamplerShapeFn");

  int64_t num_sampled;
  TF_RETURN_IF_ERROR(c->GetAttr("num_sampled", &num_sampled));
  int64_t num_true;
  TF_RETURN_IF_ERROR(c->GetAttr("num_true", &num_true));

  ShapeHandle true_classes_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &true_classes_shape));
  DimensionHandle batch_size = c->Dim(true_classes_shape, 0);

  ShapeHandle num_sampled_v = c->Vector(num_sampled);
  c->set_output(0, num_sampled_v);
  c->set_output(1, c->Matrix(batch_size, num_true));
  c->set_output(2, num_sampled_v);
  return Status::OK();
}

}  // namespace

REGISTER_OP("UniformCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful();

REGISTER_OP("LogUniformCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful();

REGISTER_OP("LearnedUnigramCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful();

REGISTER_OP("ThreadUnsafeUnigramCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful();

REGISTER_OP("FixedUnigramCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("range_max: int >= 1")
    .Attr("vocab_file: string = ''")
    .Attr("distortion: float = 1.0")
    .Attr("num_reserved_ids: int = 0")
    .Attr("num_shards: int >= 1 = 1")
    .Attr("shard: int >= 0 = 0")
    .Attr("unigrams: list(float) = []")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful();

REGISTER_OP("AllCandidateSampler")
    .Input("true_classes: int64")
    .Output("sampled_candidates: int64")
    .Output("true_expected_count: float")
    .Output("sampled_expected_count: float")
    .Attr("num_true: int >= 1")
    .Attr("num_sampled: int >= 1")
    .Attr("unique: bool")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn(CandidateSamplerShapeFn)
    .SetIsStateful();

REGISTER_OP("ComputeAccidentalHits")
    .Input("true_classes: int64")
    .Input("sampled_candidates: int64")
    .Output("indices: int32")
    .Output("ids: int64")
    .Output("weights: float")
    .Attr("num_true: int")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      int64_t num_true;
      TF_RETURN_IF_ERROR(c->GetAttr("num_true", &num_true));

      // Validate true_classes, must be a matrix.
      ShapeHandle true_classes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &true_classes));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(true_classes, 1), num_true, &unused));
      // Validate sampled_candidates, must be a vector.
      ShapeHandle sampled_candidates;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sampled_candidates));

      // All three outputs are the same shape.
      ShapeHandle v = c->Vector(InferenceContext::kUnknownDim);
      c->set_output(0, v);
      c->set_output(1, v);
      c->set_output(2, v);
      return Status::OK();
    });

}  // namespace tensorflow
