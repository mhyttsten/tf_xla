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
class MHTracer_DTPStensorflowPScPSkernelsPSopsPSbitcastDTcc {
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
   MHTracer_DTPStensorflowPScPSkernelsPSopsPSbitcastDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSkernelsPSopsPSbitcastDTcc() {
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
#include <string>

#include "tensorflow/c/ops.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

static void ComputeNewShape(TF_ShapeInferenceContext* ctx,
                            TF_ShapeHandle* shape, TF_DataType input_type,
                            TF_DataType output_type, TF_Status* status) {
  size_t input_type_size = TF_DataTypeSize(input_type);
  size_t output_type_size = TF_DataTypeSize(output_type);

  if (input_type_size == 0 || output_type_size == 0) {
    std::ostringstream err;
    err << "Cannot bitcast type " << input_type << " to " << output_type
        << " because one of the type sizes is zero";
    TF_SetStatus(status, TF_INVALID_ARGUMENT, err.str().c_str());
    return;
  }

  TF_SetStatus(status, TF_OK, "");
  if (input_type_size < output_type_size) {
    TF_ShapeInferenceContextWithRankAtLeast(ctx, shape, 1, shape, status);

    if (TF_GetCode(status) == TF_OK) {
      TF_DimensionHandle* last_dim = TF_NewDimensionHandle();
      size_t divisor_val = output_type_size / input_type_size;
      TF_ShapeInferenceContextDim(ctx, shape, -1, last_dim);
      if (!TF_DimensionHandleValueKnown(last_dim) ||
          TF_DimensionHandleValue(last_dim) == divisor_val) {
        TF_ShapeInferenceContextSubshape(ctx, shape, 0, -1, shape, status);
      } else {
        std::ostringstream err;
        err << "Cannot bitcast from " << input_type << " to " << output_type
            << " due to shape. " << TF_DimensionHandleValue(last_dim)
            << " does not match " << divisor_val;
        TF_SetStatus(status, TF_INVALID_ARGUMENT, err.str().c_str());
      }
      TF_DeleteDimensionHandle(last_dim);
    }
  } else if (input_type_size > output_type_size) {
    // Input type size is larger than output type size.
    size_t divisor_val = input_type_size / output_type_size;
    TF_ShapeHandle* extension =
        TF_ShapeInferenceContextVectorFromSize(ctx, divisor_val);
    TF_ShapeInferenceContextConcatenateShapes(ctx, shape, extension, shape,
                                              status);
    TF_DeleteShapeHandle(extension);
  }
}

static void bitcast_shape_inference_fn(TF_ShapeInferenceContext* ctx,
                                       TF_Status* status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSkernelsPSopsPSbitcastDTcc mht_0(mht_0_v, 239, "", "./tensorflow/c/kernels/ops/bitcast.cc", "bitcast_shape_inference_fn");

  TF_ShapeHandle* result = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, result, status);
  if (TF_GetCode(status) == TF_OK &&
      !TF_ShapeInferenceContextRankKnown(ctx, result)) {
    TF_ShapeInferenceContextSetUnknownShape(ctx, status);
    TF_DeleteShapeHandle(result);
    return;
  }

  // Find the size of the input and output data types.
  TF_DataType input_type;
  TF_DataType output_type;

  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContext_GetAttrType(ctx, "T", &input_type, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContext_GetAttrType(ctx, "type", &output_type, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    ComputeNewShape(ctx, result, input_type, output_type, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, result, status);
  }
  TF_DeleteShapeHandle(result);
}

void RegisterBitcastOp() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSkernelsPSopsPSbitcastDTcc mht_1(mht_1_v, 274, "", "./tensorflow/c/kernels/ops/bitcast.cc", "RegisterBitcastOp");

  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("Bitcast");
  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: type");
  TF_OpDefinitionBuilderAddAttr(
      op_builder,
      "T: {bfloat16, half, float, double, int64, int32, uint8, uint16, "
      "uint32, uint64, int8, int16, complex64, complex128, qint8, quint8, "
      "qint16, quint16, qint32}");
  TF_OpDefinitionBuilderAddAttr(
      op_builder,
      "type: {bfloat16, half, float, double, int64, int32, uint8, uint16, "
      "uint32, uint64, int8, int16, complex64, complex128, qint8, quint8, "
      "qint16, quint16, qint32}");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &bitcast_shape_inference_fn);

  TF_RegisterOpDefinition(op_builder, status);
  CHECK_EQ(TF_GetCode(status), TF_OK)
      << "Bitcast op registration failed: " << TF_Message(status);
  TF_DeleteStatus(status);
}

TF_ATTRIBUTE_UNUSED static bool IsBitcastOpRegistered = []() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSkernelsPSopsPSbitcastDTcc mht_2(mht_2_v, 302, "", "./tensorflow/c/kernels/ops/bitcast.cc", "lambda");

  if (SHOULD_REGISTER_OP("Bitcast")) {
    RegisterBitcastOp();
  }
  return true;
}();
