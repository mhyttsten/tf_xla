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
class MHTracer_DTPStensorflowPScPSc_test_utilDTcc {
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
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSc_test_utilDTcc() {
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

#include "tensorflow/c/c_test_util.h"

#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/public/session_options.h"

using tensorflow::GraphDef;
using tensorflow::NodeDef;

static void BoolDeallocator(void* data, size_t, void* arg) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_0(mht_0_v, 198, "", "./tensorflow/c/c_test_util.cc", "BoolDeallocator");

  delete[] static_cast<bool*>(data);
}

static void Int32Deallocator(void* data, size_t, void* arg) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_1(mht_1_v, 205, "", "./tensorflow/c/c_test_util.cc", "Int32Deallocator");

  delete[] static_cast<int32_t*>(data);
}

static void DoubleDeallocator(void* data, size_t, void* arg) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_2(mht_2_v, 212, "", "./tensorflow/c/c_test_util.cc", "DoubleDeallocator");

  delete[] static_cast<double*>(data);
}

static void FloatDeallocator(void* data, size_t, void* arg) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_3(mht_3_v, 219, "", "./tensorflow/c/c_test_util.cc", "FloatDeallocator");

  delete[] static_cast<float*>(data);
}

TF_Tensor* BoolTensor(bool v) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_4(mht_4_v, 226, "", "./tensorflow/c/c_test_util.cc", "BoolTensor");

  const int num_bytes = sizeof(bool);
  bool* values = new bool[1];
  values[0] = v;
  return TF_NewTensor(TF_BOOL, nullptr, 0, values, num_bytes, &BoolDeallocator,
                      nullptr);
}

TF_Tensor* Int8Tensor(const int64_t* dims, int num_dims, const char* values) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("values: \"" + (values == nullptr ? std::string("nullptr") : std::string((char*)values)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_5(mht_5_v, 238, "", "./tensorflow/c/c_test_util.cc", "Int8Tensor");

  int64_t num_values = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_values *= dims[i];
  }
  TF_Tensor* t =
      TF_AllocateTensor(TF_INT8, dims, num_dims, sizeof(char) * num_values);
  memcpy(TF_TensorData(t), values, sizeof(char) * num_values);
  return t;
}

TF_Tensor* Int32Tensor(const int64_t* dims, int num_dims,
                       const int32_t* values) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_6(mht_6_v, 253, "", "./tensorflow/c/c_test_util.cc", "Int32Tensor");

  int64_t num_values = 1;
  for (int i = 0; i < num_dims; ++i) {
    num_values *= dims[i];
  }
  TF_Tensor* t =
      TF_AllocateTensor(TF_INT32, dims, num_dims, sizeof(int32_t) * num_values);
  memcpy(TF_TensorData(t), values, sizeof(int32_t) * num_values);
  return t;
}

TF_Tensor* Int32Tensor(const std::vector<int32_t>& values) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_7(mht_7_v, 267, "", "./tensorflow/c/c_test_util.cc", "Int32Tensor");

  int64_t dims = values.size();
  return Int32Tensor(&dims, 1, values.data());
}

TF_Tensor* Int32Tensor(int32_t v) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_8(mht_8_v, 275, "", "./tensorflow/c/c_test_util.cc", "Int32Tensor");

  const int num_bytes = sizeof(int32_t);
  int32_t* values = new int32_t[1];
  values[0] = v;
  return TF_NewTensor(TF_INT32, nullptr, 0, values, num_bytes,
                      &Int32Deallocator, nullptr);
}

TF_Tensor* DoubleTensor(double v) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_9(mht_9_v, 286, "", "./tensorflow/c/c_test_util.cc", "DoubleTensor");

  const int num_bytes = sizeof(double);
  double* values = new double[1];
  values[0] = v;
  return TF_NewTensor(TF_DOUBLE, nullptr, 0, values, num_bytes,
                      &DoubleDeallocator, nullptr);
}

TF_Tensor* FloatTensor(float v) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_10(mht_10_v, 297, "", "./tensorflow/c/c_test_util.cc", "FloatTensor");

  const int num_bytes = sizeof(float);
  float* values = new float[1];
  values[0] = v;
  return TF_NewTensor(TF_FLOAT, nullptr, 0, values, num_bytes,
                      &FloatDeallocator, nullptr);
}

// All the *Helper methods are used as a workaround for the restrictions that
// one cannot call ASSERT_* methods in non-void-returning functions (when
// exceptions are disabled during compilation)
void PlaceholderHelper(TF_Graph* graph, TF_Status* s, const char* name,
                       TF_DataType dtype, const std::vector<int64_t>& dims,
                       TF_Operation** op) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_11(mht_11_v, 314, "", "./tensorflow/c/c_test_util.cc", "PlaceholderHelper");

  TF_OperationDescription* desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", dtype);
  if (!dims.empty()) {
    TF_SetAttrShape(desc, "shape", dims.data(), dims.size());
  }
  *op = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_NE(*op, nullptr);
}

TF_Operation* Placeholder(TF_Graph* graph, TF_Status* s, const char* name,
                          TF_DataType dtype, const std::vector<int64_t>& dims) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_12(mht_12_v, 330, "", "./tensorflow/c/c_test_util.cc", "Placeholder");

  TF_Operation* op;
  PlaceholderHelper(graph, s, name, dtype, dims, &op);
  return op;
}

void ConstHelper(TF_Tensor* t, TF_Graph* graph, TF_Status* s, const char* name,
                 TF_Operation** op) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_13(mht_13_v, 341, "", "./tensorflow/c/c_test_util.cc", "ConstHelper");

  TF_OperationDescription* desc = TF_NewOperation(graph, "Const", name);
  TF_SetAttrTensor(desc, "value", t, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_SetAttrType(desc, "dtype", TF_TensorType(t));
  *op = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_NE(*op, nullptr);
}

TF_Operation* Const(TF_Tensor* t, TF_Graph* graph, TF_Status* s,
                    const char* name) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_14(mht_14_v, 356, "", "./tensorflow/c/c_test_util.cc", "Const");

  TF_Operation* op;
  ConstHelper(t, graph, s, name, &op);
  return op;
}

TF_Operation* ScalarConst(bool v, TF_Graph* graph, TF_Status* s,
                          const char* name) {
   std::vector<std::string> mht_15_v;
   mht_15_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_15(mht_15_v, 367, "", "./tensorflow/c/c_test_util.cc", "ScalarConst");

  unique_tensor_ptr tensor(BoolTensor(v), TF_DeleteTensor);
  return Const(tensor.get(), graph, s, name);
}

TF_Operation* ScalarConst(int32_t v, TF_Graph* graph, TF_Status* s,
                          const char* name) {
   std::vector<std::string> mht_16_v;
   mht_16_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_16(mht_16_v, 377, "", "./tensorflow/c/c_test_util.cc", "ScalarConst");

  unique_tensor_ptr tensor(Int32Tensor(v), TF_DeleteTensor);
  return Const(tensor.get(), graph, s, name);
}

TF_Operation* ScalarConst(double v, TF_Graph* graph, TF_Status* s,
                          const char* name) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_17(mht_17_v, 387, "", "./tensorflow/c/c_test_util.cc", "ScalarConst");

  unique_tensor_ptr tensor(DoubleTensor(v), TF_DeleteTensor);
  return Const(tensor.get(), graph, s, name);
}

TF_Operation* ScalarConst(float v, TF_Graph* graph, TF_Status* s,
                          const char* name) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_18(mht_18_v, 397, "", "./tensorflow/c/c_test_util.cc", "ScalarConst");

  unique_tensor_ptr tensor(FloatTensor(v), TF_DeleteTensor);
  return Const(tensor.get(), graph, s, name);
}

void AddOpHelper(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                 TF_Status* s, const char* name, TF_Operation** op,
                 bool check) {
   std::vector<std::string> mht_19_v;
   mht_19_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_19(mht_19_v, 408, "", "./tensorflow/c/c_test_util.cc", "AddOpHelper");

  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output add_inputs[2] = {{l, 0}, {r, 0}};
  TF_AddInputList(desc, add_inputs, 2);
  *op = TF_FinishOperation(desc, s);
  if (check) {
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    ASSERT_NE(*op, nullptr);
  }
}

TF_Operation* Add(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_20(mht_20_v, 424, "", "./tensorflow/c/c_test_util.cc", "Add");

  TF_Operation* op;
  AddOpHelper(l, r, graph, s, name, &op, true);
  return op;
}

TF_Operation* AddNoCheck(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                         TF_Status* s, const char* name) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_21(mht_21_v, 435, "", "./tensorflow/c/c_test_util.cc", "AddNoCheck");

  TF_Operation* op;
  AddOpHelper(l, r, graph, s, name, &op, false);
  return op;
}

TF_Operation* AddWithCtrlDependency(TF_Operation* l, TF_Operation* r,
                                    TF_Graph* graph, TF_Operation* ctrl_op,
                                    TF_Status* s, const char* name) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_22(mht_22_v, 447, "", "./tensorflow/c/c_test_util.cc", "AddWithCtrlDependency");

  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output add_inputs[2] = {{l, 0}, {r, 0}};
  TF_AddInputList(desc, add_inputs, 2);
  TF_AddControlInput(desc, ctrl_op);
  return TF_FinishOperation(desc, s);
}

// If `op_device` is non-empty, set the created op on that device.
void BinaryOpHelper(const char* op_name, TF_Operation* l, TF_Operation* r,
                    TF_Graph* graph, TF_Status* s, const char* name,
                    TF_Operation** op, const string& op_device, bool check) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   mht_23_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_23_v.push_back("op_device: \"" + op_device + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_23(mht_23_v, 464, "", "./tensorflow/c/c_test_util.cc", "BinaryOpHelper");

  TF_OperationDescription* desc = TF_NewOperation(graph, op_name, name);
  if (!op_device.empty()) {
    TF_SetDevice(desc, op_device.c_str());
  }
  TF_AddInput(desc, {l, 0});
  TF_AddInput(desc, {r, 0});
  *op = TF_FinishOperation(desc, s);
  if (check) {
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    ASSERT_NE(*op, nullptr);
  }
}

TF_Operation* MinWithDevice(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                            const string& op_device, TF_Status* s,
                            const char* name) {
   std::vector<std::string> mht_24_v;
   mht_24_v.push_back("op_device: \"" + op_device + "\"");
   mht_24_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_24(mht_24_v, 485, "", "./tensorflow/c/c_test_util.cc", "MinWithDevice");

  TF_Operation* op;
  BinaryOpHelper("Min", l, r, graph, s, name, &op, op_device, true);
  return op;
}

TF_Operation* Min(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_25(mht_25_v, 496, "", "./tensorflow/c/c_test_util.cc", "Min");

  return MinWithDevice(l, r, graph, /*op_device=*/"", s, name);
}

TF_Operation* Mul(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                  TF_Status* s, const char* name) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_26(mht_26_v, 505, "", "./tensorflow/c/c_test_util.cc", "Mul");

  TF_Operation* op;
  BinaryOpHelper("Mul", l, r, graph, s, name, &op, "", true);
  return op;
}

TF_Operation* Add(TF_Output l, TF_Output r, TF_Graph* graph, TF_Status* s,
                  const char* name) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_27(mht_27_v, 516, "", "./tensorflow/c/c_test_util.cc", "Add");

  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output inputs[2] = {l, r};
  TF_AddInputList(desc, inputs, 2);
  return TF_FinishOperation(desc, s);
}

void NegHelper(TF_Operation* n, TF_Graph* graph, TF_Status* s, const char* name,
               TF_Operation** op) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_28(mht_28_v, 528, "", "./tensorflow/c/c_test_util.cc", "NegHelper");

  TF_OperationDescription* desc = TF_NewOperation(graph, "Neg", name);
  TF_Output neg_input = {n, 0};
  TF_AddInput(desc, neg_input);
  *op = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_NE(*op, nullptr);
}

TF_Operation* Neg(TF_Operation* n, TF_Graph* graph, TF_Status* s,
                  const char* name) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_29(mht_29_v, 542, "", "./tensorflow/c/c_test_util.cc", "Neg");

  TF_Operation* op;
  NegHelper(n, graph, s, name, &op);
  return op;
}

TF_Operation* LessThan(TF_Output l, TF_Output r, TF_Graph* graph,
                       TF_Status* s) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_30(mht_30_v, 552, "", "./tensorflow/c/c_test_util.cc", "LessThan");

  TF_OperationDescription* desc = TF_NewOperation(graph, "Less", "less_than");
  TF_AddInput(desc, l);
  TF_AddInput(desc, r);
  return TF_FinishOperation(desc, s);
}

TF_Operation* RandomUniform(TF_Operation* shape, TF_DataType dtype,
                            TF_Graph* graph, TF_Status* s) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_31(mht_31_v, 563, "", "./tensorflow/c/c_test_util.cc", "RandomUniform");

  TF_OperationDescription* desc =
      TF_NewOperation(graph, "RandomUniform", "random_uniform");
  TF_AddInput(desc, {shape, 0});
  TF_SetAttrType(desc, "dtype", dtype);
  return TF_FinishOperation(desc, s);
}

void Split3Helper(TF_Operation* input, TF_Graph* graph, TF_Status* s,
                  const char* name, TF_Operation** op) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_32(mht_32_v, 576, "", "./tensorflow/c/c_test_util.cc", "Split3Helper");

  TF_Operation* zero = ScalarConst(
      0, graph, s, ::tensorflow::strings::StrCat(name, "_const0").c_str());
  TF_OperationDescription* desc = TF_NewOperation(graph, "Split", name);
  TF_AddInput(desc, {zero, 0});
  TF_AddInput(desc, {input, 0});
  TF_SetAttrInt(desc, "num_split", 3);
  TF_SetAttrType(desc, "T", TF_INT32);
  // Set device to CPU since there is no version of split for int32 on GPU
  // TODO(iga): Convert all these helpers and tests to use floats because
  // they are usually available on GPUs. After doing this, remove TF_SetDevice
  // call in c_api_function_test.cc
  TF_SetDevice(desc, "/cpu:0");
  *op = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_NE(*op, nullptr);
}

TF_Operation* Split3(TF_Operation* input, TF_Graph* graph, TF_Status* s,
                     const char* name) {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_33(mht_33_v, 599, "", "./tensorflow/c/c_test_util.cc", "Split3");

  TF_Operation* op;
  Split3Helper(input, graph, s, name, &op);
  return op;
}

bool IsPlaceholder(const tensorflow::NodeDef& node_def) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_34(mht_34_v, 608, "", "./tensorflow/c/c_test_util.cc", "IsPlaceholder");

  if (node_def.op() != "Placeholder" || node_def.name() != "feed") {
    return false;
  }
  bool found_dtype = false;
  bool found_shape = false;
  for (const auto& attr : node_def.attr()) {
    if (attr.first == "dtype") {
      if (attr.second.type() == tensorflow::DT_INT32) {
        found_dtype = true;
      } else {
        return false;
      }
    } else if (attr.first == "shape") {
      found_shape = true;
    }
  }
  return found_dtype && found_shape;
}

bool IsScalarConst(const tensorflow::NodeDef& node_def, int v) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_35(mht_35_v, 631, "", "./tensorflow/c/c_test_util.cc", "IsScalarConst");

  if (node_def.op() != "Const" || node_def.name() != "scalar") {
    return false;
  }
  bool found_dtype = false;
  bool found_value = false;
  for (const auto& attr : node_def.attr()) {
    if (attr.first == "dtype") {
      if (attr.second.type() == tensorflow::DT_INT32) {
        found_dtype = true;
      } else {
        return false;
      }
    } else if (attr.first == "value") {
      if (attr.second.has_tensor() &&
          attr.second.tensor().int_val_size() == 1 &&
          attr.second.tensor().int_val(0) == v) {
        found_value = true;
      } else {
        return false;
      }
    }
  }
  return found_dtype && found_value;
}

bool IsAddN(const tensorflow::NodeDef& node_def, int n) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_36(mht_36_v, 660, "", "./tensorflow/c/c_test_util.cc", "IsAddN");

  if (node_def.op() != "AddN" || node_def.name() != "add" ||
      node_def.input_size() != n) {
    return false;
  }
  bool found_t = false;
  bool found_n = false;
  for (const auto& attr : node_def.attr()) {
    if (attr.first == "T") {
      if (attr.second.type() == tensorflow::DT_INT32) {
        found_t = true;
      } else {
        return false;
      }
    } else if (attr.first == "N") {
      if (attr.second.i() == n) {
        found_n = true;
      } else {
        return false;
      }
    }
  }
  return found_t && found_n;
}

bool IsNeg(const tensorflow::NodeDef& node_def, const string& input) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_37(mht_37_v, 689, "", "./tensorflow/c/c_test_util.cc", "IsNeg");

  return node_def.op() == "Neg" && node_def.name() == "neg" &&
         node_def.input_size() == 1 && node_def.input(0) == input;
}

bool GetGraphDef(TF_Graph* graph, tensorflow::GraphDef* graph_def) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_38(mht_38_v, 697, "", "./tensorflow/c/c_test_util.cc", "GetGraphDef");

  TF_Status* s = TF_NewStatus();
  TF_Buffer* buffer = TF_NewBuffer();
  TF_GraphToGraphDef(graph, buffer, s);
  bool ret = TF_GetCode(s) == TF_OK;
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  if (ret) ret = graph_def->ParseFromArray(buffer->data, buffer->length);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(s);
  return ret;
}

bool GetNodeDef(TF_Operation* oper, tensorflow::NodeDef* node_def) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_39(mht_39_v, 712, "", "./tensorflow/c/c_test_util.cc", "GetNodeDef");

  TF_Status* s = TF_NewStatus();
  TF_Buffer* buffer = TF_NewBuffer();
  TF_OperationToNodeDef(oper, buffer, s);
  bool ret = TF_GetCode(s) == TF_OK;
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  if (ret) ret = node_def->ParseFromArray(buffer->data, buffer->length);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(s);
  return ret;
}

bool GetFunctionDef(TF_Function* func, tensorflow::FunctionDef* func_def) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_40(mht_40_v, 727, "", "./tensorflow/c/c_test_util.cc", "GetFunctionDef");

  TF_Status* s = TF_NewStatus();
  TF_Buffer* buffer = TF_NewBuffer();
  TF_FunctionToFunctionDef(func, buffer, s);
  bool ret = TF_GetCode(s) == TF_OK;
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  if (ret) ret = func_def->ParseFromArray(buffer->data, buffer->length);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(s);
  return ret;
}

bool GetAttrValue(TF_Operation* oper, const char* attr_name,
                  tensorflow::AttrValue* attr_value, TF_Status* s) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("attr_name: \"" + (attr_name == nullptr ? std::string("nullptr") : std::string((char*)attr_name)) + "\"");
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_41(mht_41_v, 744, "", "./tensorflow/c/c_test_util.cc", "GetAttrValue");

  TF_Buffer* buffer = TF_NewBuffer();
  TF_OperationGetAttrValueProto(oper, attr_name, buffer, s);
  bool ret = TF_GetCode(s) == TF_OK;
  if (ret) ret = attr_value->ParseFromArray(buffer->data, buffer->length);
  TF_DeleteBuffer(buffer);
  return ret;
}

std::vector<std::pair<string, string>> GetGradDefs(
    const tensorflow::GraphDef& graph_def) {
  std::vector<std::pair<string, string>> grads;
  for (const tensorflow::GradientDef& grad : graph_def.library().gradient()) {
    grads.emplace_back(grad.function_name(), grad.gradient_func());
  }
  std::sort(grads.begin(), grads.end());
  return grads;
}

std::vector<string> GetFuncNames(const tensorflow::GraphDef& graph_def) {
  std::vector<string> names;
  auto functions = graph_def.library().function();
  names.reserve(functions.size());
  for (const tensorflow::FunctionDef& func : functions) {
    names.push_back(func.signature().name());
  }
  std::sort(names.begin(), names.end());
  return names;
}

CSession::CSession(TF_Graph* graph, TF_Status* s, bool use_XLA) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_42(mht_42_v, 777, "", "./tensorflow/c/c_test_util.cc", "CSession::CSession");

  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_EnableXLACompilation(opts, use_XLA);
  session_ = TF_NewSession(graph, opts, s);
  TF_DeleteSessionOptions(opts);
}

CSession::CSession(TF_Session* session) : session_(session) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_43(mht_43_v, 787, "", "./tensorflow/c/c_test_util.cc", "CSession::CSession");
}

CSession::~CSession() {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_44(mht_44_v, 792, "", "./tensorflow/c/c_test_util.cc", "CSession::~CSession");

  TF_Status* s = TF_NewStatus();
  CloseAndDelete(s);
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_DeleteStatus(s);
}

void CSession::SetInputs(
    std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_45(mht_45_v, 803, "", "./tensorflow/c/c_test_util.cc", "CSession::SetInputs");

  DeleteInputValues();
  inputs_.clear();
  for (const auto& p : inputs) {
    inputs_.emplace_back(TF_Output{p.first, 0});
    input_values_.emplace_back(p.second);
  }
}

void CSession::SetOutputs(std::initializer_list<TF_Operation*> outputs) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_46(mht_46_v, 815, "", "./tensorflow/c/c_test_util.cc", "CSession::SetOutputs");

  ResetOutputValues();
  outputs_.clear();
  for (TF_Operation* o : outputs) {
    outputs_.emplace_back(TF_Output{o, 0});
  }
  output_values_.resize(outputs_.size());
}

void CSession::SetOutputs(const std::vector<TF_Output>& outputs) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_47(mht_47_v, 827, "", "./tensorflow/c/c_test_util.cc", "CSession::SetOutputs");

  ResetOutputValues();
  outputs_ = outputs;
  output_values_.resize(outputs_.size());
}

void CSession::SetTargets(std::initializer_list<TF_Operation*> targets) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_48(mht_48_v, 836, "", "./tensorflow/c/c_test_util.cc", "CSession::SetTargets");

  targets_.clear();
  for (TF_Operation* t : targets) {
    targets_.emplace_back(t);
  }
}

void CSession::Run(TF_Status* s) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_49(mht_49_v, 846, "", "./tensorflow/c/c_test_util.cc", "CSession::Run");

  if (inputs_.size() != input_values_.size()) {
    ADD_FAILURE() << "Call SetInputs() before Run()";
    return;
  }
  ResetOutputValues();
  output_values_.resize(outputs_.size(), nullptr);

  const TF_Output* inputs_ptr = inputs_.empty() ? nullptr : &inputs_[0];
  TF_Tensor* const* input_values_ptr =
      input_values_.empty() ? nullptr : &input_values_[0];

  const TF_Output* outputs_ptr = outputs_.empty() ? nullptr : &outputs_[0];
  TF_Tensor** output_values_ptr =
      output_values_.empty() ? nullptr : &output_values_[0];

  TF_Operation* const* targets_ptr = targets_.empty() ? nullptr : &targets_[0];

  TF_SessionRun(session_, nullptr, inputs_ptr, input_values_ptr, inputs_.size(),
                outputs_ptr, output_values_ptr, outputs_.size(), targets_ptr,
                targets_.size(), nullptr, s);

  DeleteInputValues();
}

void CSession::CloseAndDelete(TF_Status* s) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_50(mht_50_v, 874, "", "./tensorflow/c/c_test_util.cc", "CSession::CloseAndDelete");

  DeleteInputValues();
  ResetOutputValues();
  if (session_ != nullptr) {
    TF_CloseSession(session_, s);
    EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteSession(session_, s);
    session_ = nullptr;
  }
}

void CSession::DeleteInputValues() {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_51(mht_51_v, 888, "", "./tensorflow/c/c_test_util.cc", "CSession::DeleteInputValues");

  for (size_t i = 0; i < input_values_.size(); ++i) {
    TF_DeleteTensor(input_values_[i]);
  }
  input_values_.clear();
}

void CSession::ResetOutputValues() {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScPSc_test_utilDTcc mht_52(mht_52_v, 898, "", "./tensorflow/c/c_test_util.cc", "CSession::ResetOutputValues");

  for (size_t i = 0; i < output_values_.size(); ++i) {
    if (output_values_[i] != nullptr) TF_DeleteTensor(output_values_[i]);
  }
  output_values_.clear();
}
