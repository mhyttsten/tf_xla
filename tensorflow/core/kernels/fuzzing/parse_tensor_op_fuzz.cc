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
class MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSparse_tensor_op_fuzzDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSparse_tensor_op_fuzzDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSparse_tensor_op_fuzzDTcc() {
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

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

// Fuzz inputs to the serialized Tensor decoder.

class FuzzParseTensor : public FuzzSession {
  void BuildGraph(const Scope& scope) final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSparse_tensor_op_fuzzDTcc mht_0(mht_0_v, 194, "", "./tensorflow/core/kernels/fuzzing/parse_tensor_op_fuzz.cc", "BuildGraph");

    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    // The serialized proto.
    auto input = Placeholder(scope.WithOpName("input"), DT_STRING);

    (void)ParseTensor(scope.WithOpName("output"), input, DT_FLOAT);
  }

  void FuzzImpl(const uint8_t* data, size_t size) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSfuzzingPSparse_tensor_op_fuzzDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/kernels/fuzzing/parse_tensor_op_fuzz.cc", "FuzzImpl");

    // We need to be sure that we don't request too many elements (i.e., we
    // don't make ASAN OOM). In theory, a tensor shape can have arbitrary large
    // number of elements, up to the limit of the memory available to the OS.
    // However, due to the tracing done in ASAN, after 2^32 bytes of requested
    // memory we would get a crash in the fuzzer (see b/34190148). Hence, let's
    // try parsing the proto here, check that the size (if valid) is below a
    // maximum threshold (using 2^20 for convenience), and then run the
    // remainder of the fuzzer testing. Of course, this duplicates some work
    // but it's better than repeating the investigation whenever Autofuzz
    // detects another similar OOM.
    // After adding `-fsanitize=null` to ASAN (cl/317376103), the memory
    // footprint increased, so we lower the maximum threshold to 2^18.
    string as_string = string(reinterpret_cast<const char*>(data), size);
    TensorProto proto;
    if (!ParseProtoUnlimited(&proto, as_string)) {
      LOG(WARNING) << "Unable to parse proto of tensor\n";
      return;
    }
    if (!TensorShape::IsValid(proto.tensor_shape())) {
      LOG(WARNING) << "Invalid tensor shape\n";
      return;
    }
    TensorShape shape(proto.tensor_shape());
    const int64_t num_elements = shape.num_elements();
    const int64_t max_num_elements = 1 << 18;
    if (num_elements > max_num_elements) {
      LOG(WARNING) << "Requiring a tensor with too many elements\n";
      return;
    }

    // Now we can do the actual fuzz implementation
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<tstring>()() = as_string;
    RunInputs({{"input", input_tensor}});
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzParseTensor);

}  // end namespace fuzzing
}  // end namespace tensorflow
