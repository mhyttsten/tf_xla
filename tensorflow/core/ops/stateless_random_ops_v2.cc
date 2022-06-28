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
class MHTracer_DTPStensorflowPScorePSopsPSstateless_random_ops_v2DTcc {
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
   MHTracer_DTPStensorflowPScorePSopsPSstateless_random_ops_v2DTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSopsPSstateless_random_ops_v2DTcc() {
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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/rng_alg.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

static Status StatelessShapeV2(InferenceContext* c) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSopsPSstateless_random_ops_v2DTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/ops/stateless_random_ops_v2.cc", "StatelessShapeV2");

  // Check key and counter shapes
  ShapeHandle key;
  ShapeHandle counter;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &key));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &counter));
  shape_inference::ShapeHandle unused_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused_shape));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(key, 0), RNG_KEY_SIZE, &unused));

  // Set output shape
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
  c->set_output(0, out);
  return Status::OK();
}

#define REGISTER_STATELESS_OP(name)                           \
  REGISTER_OP(name)                                           \
      .Input("shape: Tshape")                                 \
      .Input("key: uint64")                                   \
      .Input("counter: uint64")                               \
      .Input("alg: int32")                                    \
      .Output("output: dtype")                                \
      .Attr("dtype: {half,bfloat16,float,double} = DT_FLOAT") \
      .Attr("Tshape: {int32, int64} = DT_INT32")              \
      .SetShapeFn(StatelessShapeV2)

REGISTER_STATELESS_OP("StatelessRandomUniformV2");
REGISTER_STATELESS_OP("StatelessRandomNormalV2");
REGISTER_STATELESS_OP("StatelessTruncatedNormalV2");

#undef REGISTER_STATELESS_OP

REGISTER_OP("StatelessRandomUniformIntV2")
    .Input("shape: Tshape")
    .Input("key: uint64")
    .Input("counter: uint64")
    .Input("alg: int32")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64, uint32, uint64}")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      Status s = c->WithRank(c->input(4), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "minval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(4)));
      }
      s = c->WithRank(c->input(5), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "maxval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(5)));
      }
      return StatelessShapeV2(c);
    });

REGISTER_OP("StatelessRandomUniformFullIntV2")
    .Input("shape: Tshape")
    .Input("key: uint64")
    .Input("counter: uint64")
    .Input("alg: int32")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64, uint32, uint64} = DT_UINT64")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .SetShapeFn(StatelessShapeV2);

REGISTER_OP("StatelessRandomGetKeyCounterAlg")
    .Input("seed: Tseed")
    .Output("key: uint64")
    .Output("counter: uint64")
    .Output("alg: int32")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      // Check seed shape
      ShapeHandle seed;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &seed));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(seed, 0), 2, &unused));

      // Set output shapes
      c->set_output(0, c->MakeShape({RNG_KEY_SIZE}));
      c->set_output(1, c->MakeShape({RNG_MAX_COUNTER_SIZE}));
      c->set_output(2, c->MakeShape({}));
      return Status::OK();
    });

REGISTER_OP("StatelessRandomGetKeyCounter")
    .Input("seed: Tseed")
    .Output("key: uint64")
    .Output("counter: uint64")
    .Attr("Tseed: {int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      // Check seed shape
      ShapeHandle seed;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &seed));
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(seed, 0), 2, &unused));

      // Set output shapes
      c->set_output(0, c->MakeShape({RNG_KEY_SIZE}));
      c->set_output(1, c->MakeShape({RNG_MAX_COUNTER_SIZE}));
      return Status::OK();
    });

REGISTER_OP("StatelessRandomGetAlg")
    .Output("alg: int32")
    .SetIsStateful()  // because outputs depend on device
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->MakeShape({}));
      return Status::OK();
    });

}  // namespace tensorflow
