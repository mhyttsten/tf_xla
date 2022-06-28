/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This module provides helper functions for testing the interaction between
// control flow ops and subgraphs.
// For convenience, we mostly only use `kTfLiteInt32` in this module.

#ifndef TENSORFLOW_LITE_KERNELS_SUBGRAPH_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_SUBGRAPH_TEST_UTIL_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTh() {
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


#include <stdint.h>

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_test_util.h"

namespace tflite {
namespace subgraph_test_util {

class SubgraphBuilder {
 public:
  ~SubgraphBuilder();

  // Build a subgraph with a single Add op.
  // 2 inputs. 1 output.
  void BuildAddSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single Mul op.
  // 2 inputs. 1 output.
  void BuildMulSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single Pad op.
  // 2 inputs. 1 output.
  void BuildPadSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single If op.
  // 3 inputs:
  //   The 1st input is condition with boolean type.
  //   The 2nd and 3rd inputs are feed input the branch subgraphs.
  // 1 output.
  void BuildIfSubgraph(Subgraph* subgraph);

  // Build a subgraph with a single Less op.
  // The subgraph is used as the condition subgraph for testing `While` op.
  // 2 inputs:
  //   The 1st input is a counter with `kTfLiteInt32` type.
  //   The 2nd input is ignored in this subgraph.
  // 1 output with `kTfLiteBool` type.
  //   Equivalent to (input < rhs).
  void BuildLessEqualCondSubgraph(Subgraph* subgraph, int rhs);

  // An accumulate loop body subgraph. Used to produce triangle number
  // sequence. 2 inputs and 2 outputs
  //   Equivalent to (counter, value) -> (counter + 1, counter + 1 + value)
  void BuildAccumulateLoopBodySubgraph(Subgraph* subgraph);

  // A pad loop body subgraph. When used in a loop it will repeatively enlarge
  // the
  //   tensor.
  // 2 inputs and 2 outputs.
  //   Equivalent to (counter, value) -> (counter + 1, tf.pad(value, padding))
  // Note the padding is created as a constant tensor.
  void BuildPadLoopBodySubgraph(Subgraph* subgraph,
                                const std::vector<int> padding);

  // Build a subgraph with a single While op.
  // 2 inputs, 2 outputs.
  void BuildWhileSubgraph(Subgraph* subgraph);

  // Build a subgraph that assigns a random value to a variable.
  // No input/output.
  void BuildAssignRandomValueToVariableSubgraph(Subgraph* graph);

  // Build a subgraph with CallOnce op and ReadVariable op.
  // No input and 1 output.
  void BuildCallOnceAndReadVariableSubgraph(Subgraph* graph);

  // Build a subgraph with CallOnce op, ReadVariable op and Add op.
  // No input and 1 output.
  void BuildCallOnceAndReadVariablePlusOneSubgraph(Subgraph* graph);

  // Build a subgraph with a single Less op.
  // The subgraph is used as the condition subgraph for testing `While` op.
  // 3 inputs:
  //   The 1st and 2nd inputs are string tensors, which will be ignored.
  //   The 3rd input is an integner value as a counter in this subgraph.
  // 1 output with `kTfLiteBool` type.
  //   Equivalent to (int_val < rhs).
  void BuildLessEqualCondSubgraphWithDynamicTensor(Subgraph* subgraph, int rhs);

  // Build a subgraph with a single While op, which has 3 inputs and 3 outputs.
  // This subgraph is used for creating/invoking dynamic allocated tensors based
  // on string tensors.
  //   Equivalent to (str1, str2, int_val) ->
  //                 (str1, Fill(str1, int_val + 1), int_val + 1).
  void BuildBodySubgraphWithDynamicTensor(Subgraph* subgraph);

  // Build a subgraph with a single While op, that contains 3 inputs and 3
  // outputs (str1, str2, int_val).
  void BuildWhileSubgraphWithDynamicTensor(Subgraph* subgraph);

 private:
  void CreateConstantInt32Tensor(Subgraph* subgraph, int tensor_index,
                                 const std::vector<int>& shape,
                                 const std::vector<int>& data);
  std::vector<void*> buffers_;
};

class ControlFlowOpTest : public InterpreterTest {
 public:
  ControlFlowOpTest() : builder_(new SubgraphBuilder) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTh mht_0(mht_0_v, 296, "", "./tensorflow/lite/kernels/subgraph_test_util.h", "ControlFlowOpTest");
}

  ~ControlFlowOpTest() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSsubgraph_test_utilDTh mht_1(mht_1_v, 301, "", "./tensorflow/lite/kernels/subgraph_test_util.h", "~ControlFlowOpTest");

    builder_.reset();
  }

 protected:
  std::unique_ptr<SubgraphBuilder> builder_;
};

// Fill a `TfLiteTensor` with a 32-bits integer vector.
// Preconditions:
// * The tensor must have `kTfLiteInt32` type.
// * The tensor must be allocated.
// * The element count of the tensor must be equal to the length or
//   the vector.
void FillIntTensor(TfLiteTensor* tensor, const std::vector<int32_t>& data);

// Fill a `TfLiteTensor` with a string value.
// Preconditions:
// * The tensor must have `kTfLitString` type.
void FillScalarStringTensor(TfLiteTensor* tensor, const std::string& data);

// Check if the scalar string data of a tensor is as expected.
void CheckScalarStringTensor(const TfLiteTensor* tensor,
                             const std::string& data);

// Check if the shape and string data of a tensor is as expected.
void CheckStringTensor(const TfLiteTensor* tensor,
                       const std::vector<int>& shape,
                       const std::vector<std::string>& data);

// Check if the shape and int32 data of a tensor is as expected.
void CheckIntTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                    const std::vector<int32_t>& data);
// Check if the shape and bool data of a tensor is as expected.
void CheckBoolTensor(const TfLiteTensor* tensor, const std::vector<int>& shape,
                     const std::vector<bool>& data);

}  // namespace subgraph_test_util
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SUBGRAPH_TEST_UTIL_H_
