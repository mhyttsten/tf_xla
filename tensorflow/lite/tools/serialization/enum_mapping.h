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
#ifndef TENSORFLOW_LITE_TOOLS_SERIALIZATION_ENUM_MAPPING_H_
#define TENSORFLOW_LITE_TOOLS_SERIALIZATION_ENUM_MAPPING_H_
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
class MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh() {
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


#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/schema/reflection/schema_generated.h"

// TODO(aselle): Ideally extract this from the schema.

namespace tflite {

inline ActivationFunctionType TfLiteActivationToSchemaActivation(
    TfLiteFusedActivation act) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh mht_0(mht_0_v, 195, "", "./tensorflow/lite/tools/serialization/enum_mapping.h", "TfLiteActivationToSchemaActivation");

  switch (act) {
    case kTfLiteActNone:
      return ActivationFunctionType_NONE;
    case kTfLiteActRelu:
      return ActivationFunctionType_RELU;
    case kTfLiteActReluN1To1:
      return ActivationFunctionType_RELU_N1_TO_1;
    case kTfLiteActRelu6:
      return ActivationFunctionType_RELU6;
    case kTfLiteActTanh:
      return ActivationFunctionType_TANH;
    case kTfLiteActSignBit:
      return ActivationFunctionType_SIGN_BIT;
    case kTfLiteActSigmoid:
      return ActivationFunctionType_NONE;  // TODO(aselle): Add to schema
  }
  return ActivationFunctionType_NONE;
}

inline Padding TfLitePaddingToSchemaPadding(TfLitePadding padding) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh mht_1(mht_1_v, 218, "", "./tensorflow/lite/tools/serialization/enum_mapping.h", "TfLitePaddingToSchemaPadding");

  switch (padding) {
    case kTfLitePaddingUnknown:
      return Padding_SAME;  // TODO(aselle): Consider an error.
    case kTfLitePaddingSame:
      return Padding_SAME;
    case kTfLitePaddingValid:
      return Padding_VALID;
  }
  return Padding_SAME;  // TODO(aselle): Consider an error.
}

inline TensorType TfLiteTypeToSchemaType(TfLiteType type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh mht_2(mht_2_v, 233, "", "./tensorflow/lite/tools/serialization/enum_mapping.h", "TfLiteTypeToSchemaType");

  switch (type) {
    // case kTfLiteNoType: return TensorType_NONE;
    case kTfLiteNoType:
      return TensorType_FLOAT32;  // TODO(aselle): Consider an error.
    case kTfLiteFloat32:
      return TensorType_FLOAT32;
    case kTfLiteFloat16:
      return TensorType_FLOAT16;
    case kTfLiteFloat64:
      return TensorType_FLOAT64;
    case kTfLiteInt32:
      return TensorType_INT32;
    case kTfLiteUInt32:
      return TensorType_UINT32;
    case kTfLiteUInt8:
      return TensorType_UINT8;
    case kTfLiteInt8:
      return TensorType_INT8;
    case kTfLiteInt64:
      return TensorType_INT64;
    case kTfLiteUInt64:
      return TensorType_UINT64;
    case kTfLiteString:
      return TensorType_STRING;
    case kTfLiteBool:
      return TensorType_BOOL;
    case kTfLiteUInt16:
      return TensorType_UINT16;
    case kTfLiteInt16:
      return TensorType_INT16;
    case kTfLiteComplex64:
      return TensorType_COMPLEX64;
    case kTfLiteComplex128:
      return TensorType_COMPLEX128;
    case kTfLiteResource:
      return TensorType_RESOURCE;
    case kTfLiteVariant:
      return TensorType_VARIANT;
  }
  // TODO(aselle): consider an error
}

inline FullyConnectedOptionsWeightsFormat
FullyConnectedOptionsWeightsFormatToSchema(
    TfLiteFullyConnectedWeightsFormat format) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh mht_3(mht_3_v, 281, "", "./tensorflow/lite/tools/serialization/enum_mapping.h", "FullyConnectedOptionsWeightsFormatToSchema");

  switch (format) {
    case kTfLiteFullyConnectedWeightsFormatDefault:
      return FullyConnectedOptionsWeightsFormat_DEFAULT;
    case kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8:
      return FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8;
  }
}

inline LSTMKernelType LSTMKernelTypeToSchema(TfLiteLSTMKernelType type) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh mht_4(mht_4_v, 293, "", "./tensorflow/lite/tools/serialization/enum_mapping.h", "LSTMKernelTypeToSchema");

  switch (type) {
    case kTfLiteLSTMFullKernel:
      return LSTMKernelType_FULL;
    case kTfLiteLSTMBasicKernel:
      return LSTMKernelType_BASIC;
  }
}

inline LSHProjectionType LSHProjectionTypeToSchema(
    TfLiteLSHProjectionType type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh mht_5(mht_5_v, 306, "", "./tensorflow/lite/tools/serialization/enum_mapping.h", "LSHProjectionTypeToSchema");

  switch (type) {
    case kTfLiteLshProjectionUnknown:
      return LSHProjectionType_UNKNOWN;
    case kTfLiteLshProjectionSparse:
      return LSHProjectionType_SPARSE;
    case kTfLiteLshProjectionDense:
      return LSHProjectionType_DENSE;
  }
}

inline MirrorPadMode MirrorPaddingModeToSchema(TfLiteMirrorPaddingMode mode) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh mht_6(mht_6_v, 320, "", "./tensorflow/lite/tools/serialization/enum_mapping.h", "MirrorPaddingModeToSchema");

  switch (mode) {
    case kTfLiteMirrorPaddingUnknown:
      return MirrorPadMode_REFLECT;  // TODO(aselle): consider an error
    case kTfLiteMirrorPaddingReflect:
      return MirrorPadMode_REFLECT;
    case kTfLiteMirrorPaddingSymmetric:
      return MirrorPadMode_SYMMETRIC;
  }
}

inline CombinerType CombinerTypeToSchema(TfLiteCombinerType type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSenum_mappingDTh mht_7(mht_7_v, 334, "", "./tensorflow/lite/tools/serialization/enum_mapping.h", "CombinerTypeToSchema");

  switch (type) {
    case kTfLiteCombinerTypeSum:
      return CombinerType_SUM;
    case kTfLiteCombinerTypeMean:
      return CombinerType_MEAN;
    case kTfLiteCombinerTypeSqrtn:
      return CombinerType_SQRTN;
  }
}

// int

}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_SERIALIZATION_ENUM_MAPPING_H_
