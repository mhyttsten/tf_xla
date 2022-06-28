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

#ifndef TENSORFLOW_CORE_FRAMEWORK_FUNCTION_TESTLIB_H_
#define TENSORFLOW_CORE_FRAMEWORK_FUNCTION_TESTLIB_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTh() {
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


#include <string>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace test {
namespace function {

// A helper class to make AttrSlice from initializer lists
class Attrs {
 public:
  Attrs(const std::initializer_list<  // NOLINT(runtime/explicit)
        std::pair<string, FunctionDefHelper::AttrValueWrapper>>& attrs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTh mht_0(mht_0_v, 206, "", "./tensorflow/core/framework/function_testlib.h", "Attrs");

    for (const auto& aval : attrs) {
      map_.insert({aval.first, aval.second.proto});
    }
  }

  Attrs(
      const std::vector<std::pair<string, FunctionDefHelper::AttrValueWrapper>>&
          attrs) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTh mht_1(mht_1_v, 217, "", "./tensorflow/core/framework/function_testlib.h", "Attrs");

    for (const auto& aval : attrs) {
      map_.insert({aval.first, aval.second.proto});
    }
  }

  operator AttrSlice() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTh mht_2(mht_2_v, 226, "", "./tensorflow/core/framework/function_testlib.h", "AttrSlice");
 return AttrSlice(&map_); }  // NOLINT(runtime/explicit)

 private:
  AttrValueMap map_;
};

// Helper to construct a NodeDef.
NodeDef NDef(
    StringPiece name, StringPiece op, gtl::ArraySlice<string> inputs,
    gtl::ArraySlice<std::pair<string, FunctionDefHelper::AttrValueWrapper>>
        attrs = {},
    const string& device = "");

// Helper to construct a GraphDef proto.
GraphDef GDef(gtl::ArraySlice<NodeDef> nodes,
              gtl::ArraySlice<FunctionDef> funcs = {});

// For testing convenience, we provide a few simple functions that can
// be easily executed and tested.

// x: T -> x * 2.
FunctionDef XTimesTwo();

// x: T -> cpu(x * 2) + cpu(x * 3).
FunctionDef TwoDeviceTimesFive();

// x: T -> cpu(x * 2), gpu(x * 3).
FunctionDef TwoDeviceMult();

// cpu(x): T, gpu(y): T -> cpu(x * 2), gpu(y * 3).
FunctionDef TwoDeviceInputOutput();

// Function taking a list of Tensors as input.
FunctionDef FuncWithListInput();

// Function returning a list of Tensors as output.
FunctionDef FuncWithListOutput();

// x: T -> x + x.
FunctionDef XAddX();

// x: T, y: T -> x + y.
FunctionDef XAddY();

// x: T -> x * 2, where x is int32.
FunctionDef XTimesTwoInt32();

// x: T -> (x * 2) * 2.
FunctionDef XTimesFour();

// x: T -> ((x * 2) * 2) * 2.
FunctionDef XTimes16();

// w: T, x: T, b: T -> MatMul(w, x) + b
FunctionDef WXPlusB();

// x: T -> x: T, T is a type which we automatically converts to a bool.
FunctionDef NonZero();

// x: T -> bool.
FunctionDef IsZero();

// x: T -> int64
FunctionDef RandomUniform();

// x: T, y:T  -> y: T, x: T
FunctionDef Swap();

// x: T, y: T -> y: T, x: T, the body has no nodes.
FunctionDef EmptyBodySwap();

// x: float, y: resource -> y: resource, 2*x: float.
FunctionDef ResourceOutput();

// x: resource -> x: resource
FunctionDef ResourceIdentity();

// x: resource -> y: float.
FunctionDef ReadResourceVariable();

// Contains simple control flow returning the input via an Enter op.
FunctionDef ControlFlow();

// Contains malformed control flow which can't be run by the executor.
FunctionDef InvalidControlFlow();

// x: T -> x <= N.
FunctionDef LessThanOrEqualToN(int64_t N);

// x: T, y: T -> x + 1, x * y
FunctionDef XPlusOneXTimesY();

// x: T, y: T -> x <= N
FunctionDef XYXLessThanOrEqualToN(int64_t N);

// x: T -> bool
FunctionDef RandomUniformLess();

// start: int64, stop: int64, step: int64 -> y: RangeDatasetOp::Dataset
FunctionDef MakeRangeDataset();

// input_dataset: variant, batch_size: int64, drop_remainder: bool
// -> y: BatchDatasetV2::Dataset
FunctionDef MakeBatchDataset();

// input_dataset: variant, other_arguments: Targuments, f: func,
// Targuments: list(type), output_types: list(type), output_shapes: list(shape),
// use_inter_op_parallelism: bool, preserve_cardinality: bool
// -> y: MapDatasetOp::Dataset
FunctionDef MakeMapDataset(bool has_other_args);

// input_dataset: variant, count: int64 -> y: TakeDataset::Dataset
FunctionDef MakeTakeDataset();

// x: T -> y: TensorSliceDatasetOp::Dataset
FunctionDef MakeTensorSliceDataset();

// x: T -> y: T, idx: out_idx
FunctionDef Unique();

void FunctionTestSchedClosure(std::function<void()> fn);

}  // end namespace function
}  // end namespace test
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_FUNCTION_TESTLIB_H_
