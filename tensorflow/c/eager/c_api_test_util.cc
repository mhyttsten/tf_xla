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
class MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/c_api_test_util.h"

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/util/port.h"

using tensorflow::string;
using tensorflow::tstring;

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, float value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_0(mht_0_v, 203, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestScalarTensorHandle");

  float data[] = {value};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, nullptr, 0, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx,
                                         const tensorflow::tstring& value) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_1(mht_1_v, 219, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestScalarTensorHandle");

  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_STRING, nullptr, 0, status);
  tstring* data = static_cast<tstring*>(TF_TensorData(t));
  *data = value;
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, int value) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_2(mht_2_v, 234, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestScalarTensorHandle");

  int data[] = {value};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_INT32, nullptr, 0, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, bool value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_3(mht_3_v, 249, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestScalarTensorHandle");

  bool data[] = {value};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_BOOL, nullptr, 0, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* DoubleTestMatrixTensorHandle(TFE_Context* ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_4(mht_4_v, 264, "", "./tensorflow/c/eager/c_api_test_util.cc", "DoubleTestMatrixTensorHandle");

  int64_t dims[] = {2, 2};
  double data[] = {1.0, 2.0, 3.0, 4.0};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_DOUBLE, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandle(TFE_Context* ctx) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_5(mht_5_v, 281, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestMatrixTensorHandle");

  int64_t dims[] = {2, 2};
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandleWithInput(TFE_Context* ctx,
                                                  float data[], int64_t dims[],
                                                  int num_dims) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_6(mht_6_v, 300, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestMatrixTensorHandleWithInput");

  TF_Status* status = TF_NewStatus();
  TF_Tensor* t =
      TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0], num_dims, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestTensorHandleWithDimsFloat(TFE_Context* ctx, float data[],
                                                int64_t dims[], int num_dims) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_7(mht_7_v, 316, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestTensorHandleWithDimsFloat");

  TF_Status* status = TF_NewStatus();
  TF_Tensor* t =
      TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0], num_dims, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestTensorHandleWithDimsInt(TFE_Context* ctx, int data[],
                                              int64_t dims[], int num_dims) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_8(mht_8_v, 332, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestTensorHandleWithDimsInt");

  TF_Status* status = TF_NewStatus();
  TF_Tensor* t =
      TFE_AllocateHostTensor(ctx, TF_INT32, &dims[0], num_dims, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandle100x100(TFE_Context* ctx) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_9(mht_9_v, 347, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestMatrixTensorHandle100x100");

  constexpr int64_t dims[] = {100, 100};
  constexpr int num_elements = dims[0] * dims[1];
  float data[num_elements];
  for (int i = 0; i < num_elements; ++i) {
    data[i] = 1.0f;
  }
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* DoubleTestMatrixTensorHandle3X2(TFE_Context* ctx) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_10(mht_10_v, 369, "", "./tensorflow/c/eager/c_api_test_util.cc", "DoubleTestMatrixTensorHandle3X2");

  int64_t dims[] = {3, 2};
  double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandle3X2(TFE_Context* ctx) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_11(mht_11_v, 386, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestMatrixTensorHandle3X2");

  int64_t dims[] = {3, 2};
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestVariable(TFE_Context* ctx, float value,
                               const tensorflow::string& device_name) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_12(mht_12_v, 404, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestVariable");

  TF_Status* status = TF_NewStatus();
  // Create the variable handle.
  TFE_Op* op = TFE_NewOp(ctx, "VarHandleOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(op, "shape", {}, 0, status);
  TFE_OpSetAttrString(op, "container", "localhost", 0);
  TFE_OpSetAttrString(op, "shared_name", "", 0);
  if (!device_name.empty()) {
    TFE_OpSetDevice(op, device_name.c_str(), status);
  }
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_TensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  TFE_Execute(op, &var_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_DeleteOp(op);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  CHECK_EQ(1, num_retvals);

  // Assign 'value' to it.
  op = TFE_NewOp(ctx, "AssignVariableOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpAddInput(op, var_handle, status);

  // Convert 'value' to a TF_Tensor then a TFE_TensorHandle.
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> t(
      TF_AllocateTensor(TF_FLOAT, nullptr, 0, sizeof(value)), TF_DeleteTensor);
  memcpy(TF_TensorData(t.get()), &value, TF_TensorByteSize(t.get()));

  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      value_handle(TFE_NewTensorHandle(t.get(), status),
                   TFE_DeleteTensorHandle);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_OpAddInput(op, value_handle.get(), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  num_retvals = 0;
  TFE_Execute(op, nullptr, &num_retvals, status);
  TFE_DeleteOp(op);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  CHECK_EQ(0, num_retvals);

  TF_DeleteStatus(status);

  return var_handle;
}

TFE_Op* AddOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_13(mht_13_v, 458, "", "./tensorflow/c/eager/c_api_test_util.cc", "AddOp");

  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "AddV2", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, b, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_Op* MatMulOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_14(mht_14_v, 476, "", "./tensorflow/c/eager/c_api_test_util.cc", "MatMulOp");

  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "MatMul", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, b, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_Op* IdentityOp(TFE_Context* ctx, TFE_TensorHandle* a) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_15(mht_15_v, 494, "", "./tensorflow/c/eager/c_api_test_util.cc", "IdentityOp");

  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "Identity", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_Op* ShapeOp(TFE_Context* ctx, TFE_TensorHandle* a) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_16(mht_16_v, 510, "", "./tensorflow/c/eager/c_api_test_util.cc", "ShapeOp");

  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "Shape", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_TensorHandle* TestAxisTensorHandle(TFE_Context* ctx) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_17(mht_17_v, 526, "", "./tensorflow/c/eager/c_api_test_util.cc", "TestAxisTensorHandle");

  int64_t dims[] = {1};
  int data[] = {1};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_INT32, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_Op* MinOp(TFE_Context* ctx, TFE_TensorHandle* input,
              TFE_TensorHandle* axis) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_18(mht_18_v, 544, "", "./tensorflow/c/eager/c_api_test_util.cc", "MinOp");

  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "Min", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, input, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, axis, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpSetAttrBool(op, "keep_dims", 1);
  TFE_OpSetAttrType(op, "Tidx", TF_INT32);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(input));

  return op;
}

TFE_Op* AllReduceOp(TFE_Context* ctx, TFE_TensorHandle* in, int group_size) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_19(mht_19_v, 564, "", "./tensorflow/c/eager/c_api_test_util.cc", "AllReduceOp");

  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "CollectiveReduce", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, in, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);

  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(in));
  TFE_OpSetAttrInt(op, "group_size", group_size);
  TFE_OpSetAttrInt(op, "group_key", 123);
  TFE_OpSetAttrInt(op, "instance_key", 456);
  TFE_OpSetAttrString(op, "merge_op", "Add", 3);
  TFE_OpSetAttrString(op, "final_op", "Id", 2);
  std::vector<int64_t> subdiv_offsets;
  TFE_OpSetAttrIntList(op, "subdiv_offsets", subdiv_offsets.data(),
                       subdiv_offsets.size());

  return op;
}

TFE_Op* SendOp(TFE_Context* ctx, TFE_TensorHandle* in,
               const std::string& op_name, const std::string& send_device,
               const std::string& recv_device,
               tensorflow::uint64 send_device_incarnation) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("op_name: \"" + op_name + "\"");
   mht_20_v.push_back("send_device: \"" + send_device + "\"");
   mht_20_v.push_back("recv_device: \"" + recv_device + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_20(mht_20_v, 595, "", "./tensorflow/c/eager/c_api_test_util.cc", "SendOp");

  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(ctx, op_name.c_str(), status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, in, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);

  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(in));
  TFE_OpSetAttrString(op, "tensor_name", "dummy", 5);
  TFE_OpSetAttrString(op, "send_device", send_device.c_str(),
                      send_device.size());
  TFE_OpSetAttrString(op, "recv_device", recv_device.c_str(),
                      recv_device.size());
  TFE_OpSetAttrInt(op, "send_device_incarnation", send_device_incarnation);

  return op;
}

TFE_Op* RecvOp(TFE_Context* ctx, const std::string& op_name,
               const std::string& send_device, const std::string& recv_device,
               tensorflow::uint64 send_device_incarnation) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("op_name: \"" + op_name + "\"");
   mht_21_v.push_back("send_device: \"" + send_device + "\"");
   mht_21_v.push_back("recv_device: \"" + recv_device + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_21(mht_21_v, 622, "", "./tensorflow/c/eager/c_api_test_util.cc", "RecvOp");

  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(ctx, op_name.c_str(), status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);

  TFE_OpSetAttrType(op, "tensor_type", TF_INT32);
  TFE_OpSetAttrString(op, "tensor_name", "dummy", 5);
  TFE_OpSetAttrString(op, "send_device", send_device.c_str(),
                      send_device.size());
  TFE_OpSetAttrString(op, "recv_device", recv_device.c_str(),
                      recv_device.size());
  TFE_OpSetAttrInt(op, "send_device_incarnation", send_device_incarnation);

  return op;
}

bool GetDeviceName(TFE_Context* ctx, string* device_name,
                   const char* device_type) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("device_type: \"" + (device_type == nullptr ? std::string("nullptr") : std::string((char*)device_type)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_22(mht_22_v, 644, "", "./tensorflow/c/eager/c_api_test_util.cc", "GetDeviceName");

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const int num_devices = TF_DeviceListCount(devices);
  for (int i = 0; i < num_devices; ++i) {
    const string dev_type(TF_DeviceListType(devices, i, status.get()));
    CHECK_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
    const string dev_name(TF_DeviceListName(devices, i, status.get()));
    CHECK_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
    if (dev_type == device_type) {
      *device_name = dev_name;
      LOG(INFO) << "Found " << device_type << " device " << *device_name;
      TF_DeleteDeviceList(devices);
      return true;
    }
  }
  TF_DeleteDeviceList(devices);
  return false;
}

tensorflow::ServerDef GetServerDef(const string& job_name, int num_tasks) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_23(mht_23_v, 671, "", "./tensorflow/c/eager/c_api_test_util.cc", "GetServerDef");

  tensorflow::ServerDef server_def;
  server_def.set_protocol("grpc");
  server_def.set_job_name(job_name);
  server_def.set_task_index(0);
  tensorflow::ClusterDef* cluster_def = server_def.mutable_cluster();
  tensorflow::JobDef* job_def = cluster_def->add_job();
  job_def->set_name(job_name);
  for (int i = 0; i < num_tasks; i++) {
    int port = tensorflow::testing::PickUnusedPortOrDie();
    job_def->mutable_tasks()->insert(
        {i, tensorflow::strings::StrCat("localhost:", port)});
  }
  return server_def;
}

tensorflow::ServerDef GetServerDef(int num_tasks) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_24(mht_24_v, 690, "", "./tensorflow/c/eager/c_api_test_util.cc", "GetServerDef");

  return GetServerDef("localhost", num_tasks);
}

tensorflow::ServerDef GetMultiClientServerDef(const std::string& job_name,
                                              int num_tasks,
                                              int num_virtual_gpus) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("job_name: \"" + job_name + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_test_utilDTcc mht_25(mht_25_v, 700, "", "./tensorflow/c/eager/c_api_test_util.cc", "GetMultiClientServerDef");

  tensorflow::ServerDef server_def;
  server_def.set_protocol("grpc");
  server_def.set_job_name(job_name);
  server_def.set_task_index(0);
  tensorflow::ClusterDef* cluster_def = server_def.mutable_cluster();
  tensorflow::JobDef* job_def = cluster_def->add_job();
  job_def->set_name(job_name);
  for (int i = 0; i < num_tasks; i++) {
    int port = tensorflow::testing::PickUnusedPortOrDie();
    job_def->mutable_tasks()->insert(
        {i, tensorflow::strings::StrCat("localhost:", port)});
  }
  auto* config = server_def.mutable_default_session_config();
  config->mutable_experimental()->set_collective_group_leader(
      tensorflow::strings::StrCat("/job:", job_name, "/replica:0/task:", 0));
  auto* rewrite_options =
      config->mutable_graph_options()->mutable_rewrite_options();
  rewrite_options->set_scoped_allocator_optimization(
      tensorflow::RewriterConfig::ON);
  rewrite_options->mutable_scoped_allocator_opts()->add_enable_op(
      "CollectiveReduce");

  if ((tensorflow::IsGoogleCudaEnabled() || tensorflow::IsBuiltWithROCm()) &&
      num_virtual_gpus > 0) {
    tensorflow::GPUOptions* gpu_options =
        server_def.mutable_default_session_config()->mutable_gpu_options();
    auto virtual_devices =
        gpu_options->mutable_experimental()->add_virtual_devices();
    for (int i = 0; i < num_virtual_gpus; ++i) {
      virtual_devices->add_memory_limit_mb(200);
    }
  }
  return server_def;
}
