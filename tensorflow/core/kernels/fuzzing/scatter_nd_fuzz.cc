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
class MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSscatter_nd_fuzzDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSscatter_nd_fuzzDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSscatter_nd_fuzzDTcc() {
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

/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

class FuzzScatterNd : public FuzzSession {
  void BuildGraph(const Scope& scope) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSscatter_nd_fuzzDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/kernels/fuzzing/scatter_nd_fuzz.cc", "BuildGraph");

    auto indices =
        tensorflow::ops::Placeholder(scope.WithOpName("indices"), DT_INT32);
    auto updates =
        tensorflow::ops::Placeholder(scope.WithOpName("updates"), DT_INT32);
    auto shape =
        tensorflow::ops::Placeholder(scope.WithOpName("shape"), DT_INT32);
    (void)tensorflow::ops::ScatterNd(scope.WithOpName("output"), indices,
                                     updates, shape);
  }

  void FuzzImpl(const uint8_t* data, size_t size) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSscatter_nd_fuzzDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/kernels/fuzzing/scatter_nd_fuzz.cc", "FuzzImpl");

    // This op's runtime is heavily determined by the shape of the tensor
    // arguments and almost not at all by the values of those tensors. Hence,
    // the fuzzing data here is only used to determine the shape of the
    // arguments and the output and the data of these tensors is just a constant
    // value. Furthermore, the shape of the updates_tensor tensor is fully
    // determined by the contents of the shape_tensor and the shape of the
    // indices_tensor. Rather than using random values for the
    // updates_tensor.shape and getting most of the fuzz runs stopped in the
    // check, it's better to just create a proper update_tensor.
    if (size < 1) {
      return;
    }

    // First element of the data buffer gives the number of dimensions of the
    // shape tensor.
    size_t i;
    size_t data_ix = 0;
    size_t shape_dims = 1 + (data[data_ix++] % kMaxShapeDims);
    Tensor shape_tensor(tensorflow::DT_INT32,
                        TensorShape({static_cast<int64_t>(shape_dims)}));

    // Check that we have enough elements left for the shape tensor
    if (data_ix + shape_dims >= size) {
      return;  // not enough elements, no fuzz
    }

    // Subsequent elements give the contents of the shape tensor.
    // To not get out of memory, reduce all dimensions to at most kMaxDim
    auto flat_shape = shape_tensor.flat<int32>();
    for (i = 0; i < shape_dims; i++) {
      flat_shape(i) = data[data_ix++] % kMaxDim;
    }

    // Next, we have to fill in the indices tensor. Take the next element from
    // the buffer to represent the rank of this tensor.
    if (data_ix >= size) {
      return;
    }
    size_t indices_rank = 1 + (data[data_ix++] % kMaxIndicesRank);

    // Now, read the dimensions of the indices_tensor
    if (data_ix + indices_rank >= size) {
      return;
    }
    std::vector<int64_t> indices_dims;
    size_t num_indices = 1;
    for (i = 0; i < indices_rank; i++) {
      // Modulo kMaxDim to not request too much memory
      int64_t dim = data[data_ix++] % kMaxDim;
      num_indices *= dim;
      indices_dims.push_back(dim);
    }
    Tensor indices_tensor(tensorflow::DT_INT32, TensorShape(indices_dims));

    // Rest of the buffer is used to fill in the indices_tensor
    auto flat_indices = indices_tensor.flat<int32>();
    for (i = 0; i < num_indices && data_ix < size; i++) {
      flat_indices(i) = data[data_ix++];
    }
    for (; i < num_indices; i++) {
      flat_indices(i) = 0;  // ensure that indices_tensor has all values
    }

    // Given the values in the shape_tensor and the dimensions of the
    // indices_tensor, the shape of updates_tensor is fixed.
    num_indices = 1;
    std::vector<int64_t> updates_dims;
    for (i = 0; i < indices_rank - 1; i++) {
      updates_dims.push_back(indices_dims[i]);
      num_indices *= indices_dims[i];
    }
    int64_t last = indices_dims[indices_rank - 1];
    for (i = last; i < shape_dims; i++) {
      updates_dims.push_back(flat_shape(i));
      num_indices *= flat_shape(i);
    }
    Tensor updates_tensor(tensorflow::DT_INT32, TensorShape(updates_dims));

    // We don't care about the values in the updates_tensor, make them all be 1
    auto flat_updates = updates_tensor.flat<int32>();
    for (i = 0; i < num_indices; i++) {
      flat_updates(i) = 1;
    }

    RunInputs({{"indices", indices_tensor},
               {"updates", updates_tensor},
               {"shape", shape_tensor}});
  }

 private:
  const size_t kMaxShapeDims = 5;
  const size_t kMaxIndicesRank = 3;
  const size_t kMaxDim = 10;
};

STANDARD_TF_FUZZ_FUNCTION(FuzzScatterNd);

}  // end namespace fuzzing
}  // end namespace tensorflow
