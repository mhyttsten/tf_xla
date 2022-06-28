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
class MHTracer_DTPStensorflowPScPSopsDTcc {
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
   MHTracer_DTPStensorflowPScPSopsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSopsDTcc() {
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

#include "tensorflow/c/ops.h"

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::DataType;
using ::tensorflow::OpDef;
using ::tensorflow::OpDefBuilder;
using ::tensorflow::OpDeprecation;
using ::tensorflow::OpShapeInferenceFn;
using ::tensorflow::Set_TF_Status_from_Status;
using ::tensorflow::Status;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

TF_OpDefinitionBuilder* TF_NewOpDefinitionBuilder(const char* op_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_name: \"" + (op_name == nullptr ? std::string("nullptr") : std::string((char*)op_name)) + "\"");
   MHTracer_DTPStensorflowPScPSopsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/c/ops.cc", "TF_NewOpDefinitionBuilder");

  auto* result = new OpDefBuilder(op_name);
  return reinterpret_cast<TF_OpDefinitionBuilder*>(result);
}

void TF_DeleteOpDefinitionBuilder(TF_OpDefinitionBuilder* builder) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_1(mht_1_v, 213, "", "./tensorflow/c/ops.cc", "TF_DeleteOpDefinitionBuilder");

  delete reinterpret_cast<OpDefBuilder*>(builder);
}

void TF_OpDefinitionBuilderAddInput(TF_OpDefinitionBuilder* builder,
                                    const char* input_spec) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("input_spec: \"" + (input_spec == nullptr ? std::string("nullptr") : std::string((char*)input_spec)) + "\"");
   MHTracer_DTPStensorflowPScPSopsDTcc mht_2(mht_2_v, 222, "", "./tensorflow/c/ops.cc", "TF_OpDefinitionBuilderAddInput");

  reinterpret_cast<OpDefBuilder*>(builder)->Input(input_spec);
}

void TF_OpDefinitionBuilderAddOutput(TF_OpDefinitionBuilder* builder,
                                     const char* output_spec) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("output_spec: \"" + (output_spec == nullptr ? std::string("nullptr") : std::string((char*)output_spec)) + "\"");
   MHTracer_DTPStensorflowPScPSopsDTcc mht_3(mht_3_v, 231, "", "./tensorflow/c/ops.cc", "TF_OpDefinitionBuilderAddOutput");

  reinterpret_cast<OpDefBuilder*>(builder)->Output(output_spec);
}

#define DEFINE_BUILDER_BOOL_SETTER(func_name)                             \
  void TF_OpDefinitionBuilder##func_name(TF_OpDefinitionBuilder* builder, \
                                         bool arg_name) {                 \
    reinterpret_cast<OpDefBuilder*>(builder)->func_name();                \
  }

DEFINE_BUILDER_BOOL_SETTER(SetIsCommutative)
DEFINE_BUILDER_BOOL_SETTER(SetIsAggregate)
DEFINE_BUILDER_BOOL_SETTER(SetIsStateful)
DEFINE_BUILDER_BOOL_SETTER(SetAllowsUninitializedInput)

void TF_OpDefinitionBuilderAddAttr(TF_OpDefinitionBuilder* builder,
                                   const char* attr_spec) {
  reinterpret_cast<OpDefBuilder*>(builder)->Attr(attr_spec);
}

void TF_OpDefinitionBuilderDeprecated(TF_OpDefinitionBuilder* builder,
                                      int version, const char* explanation) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("explanation: \"" + (explanation == nullptr ? std::string("nullptr") : std::string((char*)explanation)) + "\"");
   MHTracer_DTPStensorflowPScPSopsDTcc mht_4(mht_4_v, 256, "", "./tensorflow/c/ops.cc", "TF_OpDefinitionBuilderDeprecated");

  reinterpret_cast<OpDefBuilder*>(builder)->Deprecated(version, explanation);
}

void TF_RegisterOpDefinition(TF_OpDefinitionBuilder* builder,
                             TF_Status* status) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_5(mht_5_v, 264, "", "./tensorflow/c/ops.cc", "TF_RegisterOpDefinition");

  auto* cc_builder = reinterpret_cast<OpDefBuilder*>(builder);
  TF_SetStatus(status, TF_OK, "");
  ::tensorflow::OpRegistry::Global()->Register(
      [cc_builder](::tensorflow::OpRegistrationData* op_reg_data) -> Status {
        Status result = cc_builder->Finalize(op_reg_data);
        delete cc_builder;
        return result;
      });
}

void TF_OpDefinitionBuilderSetShapeInferenceFunction(
    TF_OpDefinitionBuilder* builder,
    void (*shape_inference_func)(TF_ShapeInferenceContext* ctx,
                                 TF_Status* status)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_6(mht_6_v, 281, "", "./tensorflow/c/ops.cc", "TF_OpDefinitionBuilderSetShapeInferenceFunction");

  auto* cc_builder = reinterpret_cast<OpDefBuilder*>(builder);
  cc_builder->SetShapeFn(
      [shape_inference_func](InferenceContext* ctx) -> tensorflow::Status {
        TF_Status* c_status = TF_NewStatus();
        auto c_ctx = reinterpret_cast<TF_ShapeInferenceContext*>(ctx);
        shape_inference_func(c_ctx, c_status);
        tensorflow::Status result = ::tensorflow::StatusFromTF_Status(c_status);
        TF_DeleteStatus(c_status);
        return result;
      });
}

TF_ShapeHandle* TF_NewShapeHandle() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_7(mht_7_v, 297, "", "./tensorflow/c/ops.cc", "TF_NewShapeHandle");

  return reinterpret_cast<TF_ShapeHandle*>(new ShapeHandle);
}

TF_ShapeHandle* TF_ShapeInferenceContextScalar(TF_ShapeInferenceContext* ctx) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_8(mht_8_v, 304, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextScalar");

  auto* handle = new ShapeHandle;
  *handle = reinterpret_cast<InferenceContext*>(ctx)->Scalar();
  return reinterpret_cast<TF_ShapeHandle*>(handle);
}

TF_ShapeHandle* TF_ShapeInferenceContextVectorFromSize(
    TF_ShapeInferenceContext* ctx, size_t size) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_9(mht_9_v, 314, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextVectorFromSize");

  auto* handle = new ShapeHandle;
  *handle = reinterpret_cast<InferenceContext*>(ctx)->Vector(size);
  return reinterpret_cast<TF_ShapeHandle*>(handle);
}

void TF_ShapeInferenceContextConcatenateShapes(TF_ShapeInferenceContext* ctx,
                                               TF_ShapeHandle* first,
                                               TF_ShapeHandle* second,
                                               TF_ShapeHandle* result,
                                               TF_Status* status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_10(mht_10_v, 327, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextConcatenateShapes");

  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  Status s = cc_ctx->Concatenate(*reinterpret_cast<ShapeHandle*>(first),
                                 *reinterpret_cast<ShapeHandle*>(second),
                                 reinterpret_cast<ShapeHandle*>(result));
  Set_TF_Status_from_Status(status, s);
}

TF_DimensionHandle* TF_NewDimensionHandle() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_11(mht_11_v, 338, "", "./tensorflow/c/ops.cc", "TF_NewDimensionHandle");

  return reinterpret_cast<TF_DimensionHandle*>(new DimensionHandle);
}

int64_t TF_ShapeInferenceContextNumInputs(TF_ShapeInferenceContext* ctx) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_12(mht_12_v, 345, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextNumInputs");

  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  return cc_ctx->num_inputs();
}

void TF_ShapeInferenceContextGetInput(TF_ShapeInferenceContext* ctx, int i,
                                      TF_ShapeHandle* handle,
                                      TF_Status* status) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_13(mht_13_v, 355, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextGetInput");

  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  if (0 < i || i >= cc_ctx->num_inputs()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "input index out of range");
  }
  if (TF_GetCode(status) == TF_OK) {
    auto* cc_result = reinterpret_cast<ShapeHandle*>(handle);
    *cc_result = cc_ctx->input(i);
  }
}

int TF_ShapeInferenceContextRankKnown(TF_ShapeInferenceContext* ctx,
                                      TF_ShapeHandle* handle) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_14(mht_14_v, 371, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextRankKnown");

  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  return cc_ctx->RankKnown(*reinterpret_cast<ShapeHandle*>(handle));
}

void TF_ShapeInferenceContextSetOutput(TF_ShapeInferenceContext* ctx, int i,
                                       TF_ShapeHandle* handle,
                                       TF_Status* status) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_15(mht_15_v, 381, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextSetOutput");

  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  if (0 < i || i >= cc_ctx->num_outputs()) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT, "output index out of range");
  }
  if (TF_GetCode(status) == TF_OK) {
    cc_ctx->set_output(i, *(reinterpret_cast<ShapeHandle*>(handle)));
  }
}

void TF_DeleteShapeHandle(TF_ShapeHandle* handle) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_16(mht_16_v, 395, "", "./tensorflow/c/ops.cc", "TF_DeleteShapeHandle");

  if (handle == nullptr) {
    return;
  }

  delete reinterpret_cast<ShapeHandle*>(handle);
}

void TF_DeleteDimensionHandle(TF_DimensionHandle* handle) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_17(mht_17_v, 406, "", "./tensorflow/c/ops.cc", "TF_DeleteDimensionHandle");

  if (handle == nullptr) {
    return;
  }

  delete reinterpret_cast<DimensionHandle*>(handle);
}

#define DEFINE_TF_GETATTR(func, c_type, cc_type)                         \
  void TF_ShapeInferenceContext_GetAttr##func(                           \
      TF_ShapeInferenceContext* ctx, const char* attr_name, c_type* val, \
      TF_Status* status) {                                               \
    TF_SetStatus(status, TF_OK, "");                                     \
    cc_type v;                                                           \
    auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);             \
    Status s = cc_ctx->GetAttr(attr_name, &v);                           \
    Set_TF_Status_from_Status(status, s);                                \
    if (s.ok()) {                                                        \
      *val = static_cast<c_type>(v);                                     \
    }                                                                    \
  }

DEFINE_TF_GETATTR(Type, TF_DataType, tensorflow::DataType)

#define DEFINE_RANK_FUNC(func_name)                                        \
  void TF_ShapeInferenceContext##func_name(                                \
      TF_ShapeInferenceContext* ctx, TF_ShapeHandle* handle, int64_t rank, \
      TF_ShapeHandle* result, TF_Status* status) {                         \
    auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);               \
    auto* cc_handle = reinterpret_cast<ShapeHandle*>(handle);              \
    auto* cc_result = reinterpret_cast<ShapeHandle*>(result);              \
    Status s = cc_ctx->func_name(*cc_handle, rank, cc_result);             \
    Set_TF_Status_from_Status(status, s);                                  \
  }

DEFINE_RANK_FUNC(WithRank)
DEFINE_RANK_FUNC(WithRankAtLeast)
DEFINE_RANK_FUNC(WithRankAtMost)

int64_t TF_ShapeInferenceContextRank(TF_ShapeInferenceContext* ctx,
                                     TF_ShapeHandle* handle) {
  return reinterpret_cast<InferenceContext*>(ctx)->Rank(
      *reinterpret_cast<ShapeHandle*>(handle));
}

void TF_ShapeInferenceContextDim(TF_ShapeInferenceContext* ctx,
                                 TF_ShapeHandle* shape_handle, int64_t i,
                                 TF_DimensionHandle* result) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_18(mht_18_v, 456, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextDim");

  int64_t rank = TF_ShapeInferenceContextRank(ctx, shape_handle);
  auto* cc_result = reinterpret_cast<DimensionHandle*>(result);

  if (i < -rank || i >= rank) {
    *cc_result = DimensionHandle();
    return;
  }

  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  auto* cc_shape_handle = reinterpret_cast<ShapeHandle*>(shape_handle);
  *cc_result = cc_ctx->Dim(*cc_shape_handle, i);
}

int TF_DimensionHandleValueKnown(TF_DimensionHandle* dim_handle) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_19(mht_19_v, 473, "", "./tensorflow/c/ops.cc", "TF_DimensionHandleValueKnown");

  return InferenceContext::ValueKnown(
      *reinterpret_cast<DimensionHandle*>(dim_handle));
}

void TF_ShapeInferenceContextSetUnknownShape(TF_ShapeInferenceContext* ctx,
                                             TF_Status* status) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_20(mht_20_v, 482, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextSetUnknownShape");

  Status s = ::tensorflow::shape_inference::UnknownShape(
      reinterpret_cast<InferenceContext*>(ctx));
  Set_TF_Status_from_Status(status, s);
}

void TF_ShapeInferenceContextSubshape(TF_ShapeInferenceContext* ctx,
                                      TF_ShapeHandle* shape_handle,
                                      int64_t start, int64_t end,
                                      TF_ShapeHandle* result,
                                      TF_Status* status) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_21(mht_21_v, 495, "", "./tensorflow/c/ops.cc", "TF_ShapeInferenceContextSubshape");

  TF_SetStatus(status, TF_OK, "");
  auto* cc_ctx = reinterpret_cast<InferenceContext*>(ctx);
  auto* cc_result = reinterpret_cast<ShapeHandle*>(result);
  Status s = cc_ctx->Subshape(*reinterpret_cast<ShapeHandle*>(shape_handle),
                              start, end, cc_result);
  Set_TF_Status_from_Status(status, s);
}

int64_t TF_DimensionHandleValue(TF_DimensionHandle* dim_handle) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPSopsDTcc mht_22(mht_22_v, 507, "", "./tensorflow/c/ops.cc", "TF_DimensionHandleValue");

  return InferenceContext::Value(
      *reinterpret_cast<DimensionHandle*>(dim_handle));
}
