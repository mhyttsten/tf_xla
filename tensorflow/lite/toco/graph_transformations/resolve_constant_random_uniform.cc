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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_random_uniformDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_random_uniformDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_random_uniformDTcc() {
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
#include <algorithm>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace toco {

template <ArrayDataType Type>
bool ComputeRandomUniformArray(Model* model, RandomUniformOperator* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_random_uniformDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_random_uniform.cc", "ComputeRandomUniformArray");

  typedef tensorflow::random::UniformDistribution<
      tensorflow::random::PhiloxRandom, DataType<Type>>
      Distribution;

  // Allocate output
  auto& output_array = model->GetArray(op->outputs[0]);
  CHECK(output_array.data_type == Type);
  std::vector<DataType<Type>>& data =
      output_array.GetMutableBuffer<Type>().data;
  data.resize(RequiredBufferSizeForShape(output_array.shape()));

  // We use the same random number generator and distribution as TensorFlow to
  // produce the exact same values given the same seeds. See
  // tensorflow::functor::FillPhiloxRandomTask<Distribution, false> in
  // //third_party/tensorflow/core/kernels/random_op.cc for the implementation.
  tensorflow::random::PhiloxRandom generator(op->seed, op->seed2);
  Distribution dist;

  // The generator creates Distribution::kResultElementCount samples at a time.
  size_t offset = 0;
  size_t num_samples = Distribution::kResultElementCount;
  while (offset < data.size()) {
    const typename Distribution::ResultType samples = dist(&generator);
    std::copy(&samples[0],
              &samples[0] + std::min(num_samples, data.size() - offset),
              &data[0] + offset);
    offset += num_samples;
  }

  return true;
}

::tensorflow::Status ResolveConstantRandomUniform::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_random_uniformDTcc mht_1(mht_1_v, 236, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_random_uniform.cc", "ResolveConstantRandomUniform::Run");

  *modified = false;
  const auto it = model->operators.begin() + op_index;
  auto* base_op = it->get();
  if (base_op->type != OperatorType::kRandomUniform) {
    return ::tensorflow::Status::OK();
  }
  auto* op = static_cast<RandomUniformOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return ::tensorflow::Status::OK();
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes
    return ::tensorflow::Status::OK();
  }

  if ((op->seed == 0) && (op->seed2 == 0)) {
    LOG(WARNING) << "RandomUniform op outputting \"" << op->outputs[0]
                 << "\" is truly random (using /dev/random system entropy). "
                    "Therefore, cannot resolve as constant. Set \"seed\" or "
                    "\"seed2\" attr non-zero to fix this";
    return ::tensorflow::Status::OK();
  }

  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      if (!ComputeRandomUniformArray<ArrayDataType::kFloat>(model, op)) {
        return ::tensorflow::Status::OK();
      }
      break;
    // For future support of double or half.
    // case ArrayDataType::kDouble...
    default:
      LOG(FATAL)
          << "Unsupported data type given to RandomUniform op with output \""
          << op->outputs[0] << "\"";
      break;
  }

  DeleteOpAndArrays(model, op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
