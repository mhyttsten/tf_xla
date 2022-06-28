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
class MHTracer_DTPStensorflowPScPSkernelsPSbitcast_opDTcc {
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
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSkernelsPSbitcast_opDTcc() {
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

#include <sstream>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/macros.h"

// BitcastOp implements a bitcast kernel, creating an output tensor that shares
// the same data buffer as the input but with a different shape and/or data
// type. Its inputs are:
//
//   * the input tensor
//   * an attribute named "T" containing the TF_DataType of the input tensor
//   * an attribute named "type" containing the TF_DataType of the output tensor
//
// Given an input tensor of shape [...], if the input DataType "T" is larger
// than the output DataType "type", then the shape changes from [...]
// to [..., sizeof(T)/sizeof(type)].
//
// If "T" is smaller than "type", the operator requires that the rightmost
// dimension be equal to sizeof(type)/sizeof(T). The shape then goes from
// [..., sizeof(type)/sizeof(T)] to [...].
//
// Bitcast is implemented as a low-level cast, so machines with different endian
// orderings will give different results.
typedef struct BitcastOp {
  TF_DataType input_data_type;
  TF_DataType output_data_type;
  size_t in_size;
  size_t out_size;
} BitcastOp;

static void* BitcastOp_Create(TF_OpKernelConstruction* ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_opDTcc mht_0(mht_0_v, 221, "", "./tensorflow/c/kernels/bitcast_op.cc", "BitcastOp_Create");

  auto* kernel = new BitcastOp;

  TF_Status* s = TF_NewStatus();
  TF_OpKernelConstruction_GetAttrType(ctx, "T", &kernel->input_data_type, s);

  if (TF_GetCode(s) == TF_OK) {
    TF_OpKernelConstruction_GetAttrType(ctx, "type", &kernel->output_data_type,
                                        s);
  }

  if (TF_GetCode(s) == TF_OK) {
    kernel->in_size = TF_DataTypeSize(kernel->input_data_type);
    kernel->out_size = TF_DataTypeSize(kernel->output_data_type);

    size_t check_size = std::max(kernel->in_size, kernel->out_size) %
                        std::min(kernel->in_size, kernel->out_size);
    if (check_size != 0) {
      std::ostringstream err;
      err << "cannot convert between datatype " << kernel->input_data_type
          << " and " << kernel->output_data_type;
      TF_SetStatus(s, TF_INVALID_ARGUMENT, err.str().c_str());
    }
  }

  if (TF_GetCode(s) != TF_OK) {
    TF_OpKernelConstruction_Failure(ctx, s);
    delete kernel;
    kernel = nullptr;
  }

  TF_DeleteStatus(s);
  return kernel;
}

static void BitcastOp_Delete(void* kernel) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_opDTcc mht_1(mht_1_v, 259, "", "./tensorflow/c/kernels/bitcast_op.cc", "BitcastOp_Delete");

  delete static_cast<BitcastOp*>(kernel);
}

static void BitcastOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_opDTcc mht_2(mht_2_v, 266, "", "./tensorflow/c/kernels/bitcast_op.cc", "BitcastOp_Compute");

  auto* k = static_cast<BitcastOp*>(kernel);
  int dim_count = 0;

  TF_Tensor* tensor;
  TF_Status* status = TF_NewStatus();
  TF_GetInput(ctx, 0, &tensor, status);
  if (TF_GetCode(status) == TF_OK) {
    dim_count = TF_NumDims(tensor);
    if (!(k->in_size >= k->out_size ||
          (dim_count > 0 &&
           TF_Dim(tensor, dim_count - 1) == k->out_size / k->in_size))) {
      std::ostringstream err;
      err << "Cannot bitcast from " << k->input_data_type << " to "
          << k->output_data_type;
      TF_SetStatus(status, TF_INVALID_ARGUMENT, err.str().c_str());
    }
  }

  if (TF_GetCode(status) == TF_OK) {
    auto* dims = new int64_t[dim_count + 1];
    int new_dim_count = dim_count;
    for (int dim = 0; dim < dim_count; ++dim) {
      dims[dim] = TF_Dim(tensor, dim);
    }
    if (k->out_size < k->in_size) {
      dims[new_dim_count++] = static_cast<int64_t>(k->in_size / k->out_size);
    } else if (k->out_size > k->in_size) {
      --new_dim_count;
    }

    TF_Tensor* output = TF_AllocateTensor(k->output_data_type, dims, 0,
                                          TF_DataTypeSize(k->output_data_type));
    TF_TensorBitcastFrom(tensor, k->output_data_type, output, dims,
                         new_dim_count, status);
    if (TF_GetCode(status) == TF_OK) {
      TF_SetOutput(ctx, 0, output, status);
    }
    delete[] dims;
    TF_DeleteTensor(output);
  }

  if (TF_GetCode(status) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status);
  }
  TF_DeleteStatus(status);
  TF_DeleteTensor(tensor);
}

void RegisterBitcastOpKernel() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_opDTcc mht_3(mht_3_v, 318, "", "./tensorflow/c/kernels/bitcast_op.cc", "RegisterBitcastOpKernel");

  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder("Bitcast", tensorflow::DEVICE_CPU,
                                        &BitcastOp_Create, &BitcastOp_Compute,
                                        &BitcastOp_Delete);
    TF_RegisterKernelBuilder("BitcastOp", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering bitcast kernel";
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  {
    auto* builder = TF_NewKernelBuilder("Bitcast", tensorflow::DEVICE_GPU,
                                        &BitcastOp_Create, &BitcastOp_Compute,
                                        &BitcastOp_Delete);
    TF_RegisterKernelBuilder("BitcastOp", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering CUDA bitcast kernel";
  }
#endif

  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the bitcast kernel.
TF_ATTRIBUTE_UNUSED static bool IsBitcastOpKernelRegistered = []() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSkernelsPSbitcast_opDTcc mht_4(mht_4_v, 348, "", "./tensorflow/c/kernels/bitcast_op.cc", "lambda");

  if (SHOULD_REGISTER_OP_KERNEL("BitcastOp")) {
    RegisterBitcastOpKernel();
  }
  return true;
}();
