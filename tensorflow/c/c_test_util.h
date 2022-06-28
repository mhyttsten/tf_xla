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

#ifndef TENSORFLOW_C_C_TEST_UTIL_H_
#define TENSORFLOW_C_C_TEST_UTIL_H_
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
class MHTracer_DTPStensorflowPScPSc_test_utilDTh {
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
   MHTracer_DTPStensorflowPScPSc_test_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSc_test_utilDTh() {
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


#include "tensorflow/c/c_api.h"

#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"

using ::tensorflow::string;

typedef std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>
    unique_tensor_ptr;

TF_Tensor* BoolTensor(int32_t v);

// Create a tensor with values of type TF_INT8 provided by `values`.
TF_Tensor* Int8Tensor(const int64_t* dims, int num_dims, const char* values);

// Create a tensor with values of type TF_INT32 provided by `values`.
TF_Tensor* Int32Tensor(const int64_t* dims, int num_dims,
                       const int32_t* values);

// Create 1 dimensional tensor with values from `values`
TF_Tensor* Int32Tensor(const std::vector<int32_t>& values);

TF_Tensor* Int32Tensor(int32_t v);

TF_Tensor* DoubleTensor(double v);

TF_Tensor* FloatTensor(float v);

TF_Operation* Placeholder(TF_Graph* graph, TF_Status* s,
                          const char* name = "feed",
                          TF_DataType dtype = TF_INT32,
                          const std::vector<int64_t>& dims = {});

TF_Operation* Const(TF_Tensor* t, TF_Graph* graph, TF_Status* s,
                    const char* name = "const");

TF_Operation* ScalarConst(bool v, TF_Graph* graph, TF_Status* s,
                          const char* name = "scalar");

TF_Operation* ScalarConst(int32_t v, TF_Graph* graph, TF_Status* s,
                          const char* name = "scalar");

TF_Operation* ScalarConst(double v, TF_Graph* graph, TF_Status* s,
                          const char* name = "scalar");

TF_Operation* ScalarConst(float v, TF_Graph* graph, TF_Status* s,
                          const char* name = "scalar");

TF_Operation* Add(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name = "add");

TF_Operation* AddNoCheck(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                         TF_Status* s, const char* name = "add");

TF_Operation* AddWithCtrlDependency(TF_Operation* l, TF_Operation* r,
                                    TF_Graph* graph, TF_Operation* ctrl_op,
                                    TF_Status* s, const char* name = "add");

TF_Operation* Add(TF_Output l, TF_Output r, TF_Graph* graph, TF_Status* s,
                  const char* name = "add");

TF_Operation* Min(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name = "min");

TF_Operation* Mul(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name = "mul");

// If `op_device` is non-empty, set the created op on that device.
TF_Operation* MinWithDevice(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                            const string& op_device, TF_Status* s,
                            const char* name = "min");

TF_Operation* Neg(TF_Operation* n, TF_Graph* graph, TF_Status* s,
                  const char* name = "neg");

TF_Operation* LessThan(TF_Output l, TF_Output r, TF_Graph* graph, TF_Status* s);

TF_Operation* RandomUniform(TF_Operation* shape, TF_DataType dtype,
                            TF_Graph* graph, TF_Status* s);

// Split `input` along the first dimension into 3 tensors
TF_Operation* Split3(TF_Operation* input, TF_Graph* graph, TF_Status* s,
                     const char* name = "split3");

bool IsPlaceholder(const tensorflow::NodeDef& node_def);

bool IsScalarConst(const tensorflow::NodeDef& node_def, int v);

bool IsAddN(const tensorflow::NodeDef& node_def, int n);

bool IsNeg(const tensorflow::NodeDef& node_def, const string& input);

bool GetGraphDef(TF_Graph* graph, tensorflow::GraphDef* graph_def);

bool GetNodeDef(TF_Operation* oper, tensorflow::NodeDef* node_def);

bool GetFunctionDef(TF_Function* func, tensorflow::FunctionDef* func_def);

bool GetAttrValue(TF_Operation* oper, const char* attr_name,
                  tensorflow::AttrValue* attr_value, TF_Status* s);

// Returns a sorted vector of std::pair<function_name, gradient_func> from
// graph_def.library().gradient()
std::vector<std::pair<string, string>> GetGradDefs(
    const tensorflow::GraphDef& graph_def);

// Returns a sorted vector of names contained in `grad_def`
std::vector<string> GetFuncNames(const tensorflow::GraphDef& graph_def);

class CSession {
 public:
  CSession(TF_Graph* graph, TF_Status* s, bool use_XLA = false);
  explicit CSession(TF_Session* session);

  ~CSession();

  void SetInputs(std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs);
  void SetOutputs(std::initializer_list<TF_Operation*> outputs);
  void SetOutputs(const std::vector<TF_Output>& outputs);
  void SetTargets(std::initializer_list<TF_Operation*> targets);

  void Run(TF_Status* s);

  void CloseAndDelete(TF_Status* s);

  TF_Tensor* output_tensor(int i) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTh mht_0(mht_0_v, 318, "", "./tensorflow/c/c_test_util.h", "output_tensor");
 return output_values_[i]; }

  TF_Session* mutable_session() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTh mht_1(mht_1_v, 323, "", "./tensorflow/c/c_test_util.h", "mutable_session");
 return session_; }

 private:
  void DeleteInputValues();
  void ResetOutputValues();

  TF_Session* session_;
  std::vector<TF_Output> inputs_;
  std::vector<TF_Tensor*> input_values_;
  std::vector<TF_Output> outputs_;
  std::vector<TF_Tensor*> output_values_;
  std::vector<TF_Operation*> targets_;
};

#endif  // TENSORFLOW_C_C_TEST_UTIL_H_
