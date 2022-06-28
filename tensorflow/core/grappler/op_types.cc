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
class MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc() {
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

#include "tensorflow/core/grappler/op_types.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {

bool IsAdd(const NodeDef& node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/grappler/op_types.cc", "IsAdd");

  if (node.op() == "AddV2") {
    return true;
  }
  if (node.op() == "Add") {
    DataType type = node.attr().at("T").type();
    return type != DT_STRING;
  }
  return false;
}

bool IsAddN(const NodeDef& node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/grappler/op_types.cc", "IsAddN");
 return node.op() == "AddN"; }

bool IsAll(const NodeDef& node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/grappler/op_types.cc", "IsAll");
 return node.op() == "All"; }

bool IsAngle(const NodeDef& node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_3(mht_3_v, 223, "", "./tensorflow/core/grappler/op_types.cc", "IsAngle");
 return node.op() == "Angle"; }

bool IsAny(const NodeDef& node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_4(mht_4_v, 228, "", "./tensorflow/core/grappler/op_types.cc", "IsAny");
 return node.op() == "Any"; }

bool IsAnyDiv(const NodeDef& node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_5(mht_5_v, 233, "", "./tensorflow/core/grappler/op_types.cc", "IsAnyDiv");

  return node.op() == "RealDiv" || node.op() == "Div" || node.op() == "Xdivy" ||
         node.op() == "FloorDiv" || node.op() == "TruncateDiv";
}

bool IsAnyBatchMatMul(const NodeDef& node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_6(mht_6_v, 241, "", "./tensorflow/core/grappler/op_types.cc", "IsAnyBatchMatMul");

  return node.op() == "BatchMatMul" || node.op() == "BatchMatMulV2";
}

bool IsAnyMatMul(const NodeDef& node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_7(mht_7_v, 248, "", "./tensorflow/core/grappler/op_types.cc", "IsAnyMatMul");

  return node.op() == "MatMul" || node.op() == "SparseMatMul" ||
         IsAnyBatchMatMul(node) || IsQuantizedMatMul(node);
}

bool IsAnyMax(const NodeDef& node) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_8(mht_8_v, 256, "", "./tensorflow/core/grappler/op_types.cc", "IsAnyMax");

  const auto& op = node.op();
  return op == "Max" || op == "SegmentMax" || op == "UnsortedSegmentMax";
}

bool IsAnyMaxPool(const NodeDef& node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_9(mht_9_v, 264, "", "./tensorflow/core/grappler/op_types.cc", "IsAnyMaxPool");

  const auto& op = node.op();
  return op == "MaxPool" || op == "MaxPoolV2" || op == "MaxPool3D" ||
         op == "MaxPoolWithArgmax" || op == "FractionalMaxPool";
}

bool IsAnyMin(const NodeDef& node) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_10(mht_10_v, 273, "", "./tensorflow/core/grappler/op_types.cc", "IsAnyMin");

  const auto& op = node.op();
  return op == "Min" || op == "SegmentMin" || op == "UnsortedSegmentMin";
}

bool IsAnySparseSegmentReduction(const NodeDef& node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_11(mht_11_v, 281, "", "./tensorflow/core/grappler/op_types.cc", "IsAnySparseSegmentReduction");

  const auto& op = node.op();
  return op == "SparseSegmentSum" || op == "SparseSegmentSumWithNumSegments" ||
         op == "SparseSegmentMean" ||
         op == "SparseSegmentMeanWithNumSegments" ||
         op == "SparseSegmentSqrtN" ||
         op == "SparseSegmentSqrtNWithNumSegments";
}

bool IsApproximateEqual(const NodeDef& node) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_12(mht_12_v, 293, "", "./tensorflow/core/grappler/op_types.cc", "IsApproximateEqual");

  return node.op() == "ApproximateEqual";
}

bool IsArg(const NodeDef& node) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_13(mht_13_v, 300, "", "./tensorflow/core/grappler/op_types.cc", "IsArg");

  return node.op() == "_Arg" || node.op() == "_DeviceArg";
}

bool IsArgMax(const NodeDef& node) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_14(mht_14_v, 307, "", "./tensorflow/core/grappler/op_types.cc", "IsArgMax");
 return node.op() == "ArgMax"; }

bool IsArgMin(const NodeDef& node) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_15(mht_15_v, 312, "", "./tensorflow/core/grappler/op_types.cc", "IsArgMin");
 return node.op() == "ArgMin"; }

bool IsAvgPoolGrad(const NodeDef& node) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_16(mht_16_v, 317, "", "./tensorflow/core/grappler/op_types.cc", "IsAvgPoolGrad");
 return node.op() == "AvgPoolGrad"; }

bool IsAssign(const NodeDef& node) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_17(mht_17_v, 322, "", "./tensorflow/core/grappler/op_types.cc", "IsAssign");

  return node.op() == "Assign" || node.op() == "AssignVariableOp";
}

bool IsAssert(const NodeDef& node) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_18(mht_18_v, 329, "", "./tensorflow/core/grappler/op_types.cc", "IsAssert");
 return node.op() == "Assert"; }

bool IsAsString(const NodeDef& node) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_19(mht_19_v, 334, "", "./tensorflow/core/grappler/op_types.cc", "IsAsString");
 return node.op() == "AsString"; }

bool IsAtan2(const NodeDef& node) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_20(mht_20_v, 339, "", "./tensorflow/core/grappler/op_types.cc", "IsAtan2");
 return node.op() == "Atan2"; }

bool IsBetainc(const NodeDef& node) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_21(mht_21_v, 344, "", "./tensorflow/core/grappler/op_types.cc", "IsBetainc");
 return node.op() == "Betainc"; }

bool IsBiasAdd(const NodeDef& node) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_22(mht_22_v, 349, "", "./tensorflow/core/grappler/op_types.cc", "IsBiasAdd");

  return node.op() == "BiasAdd" || node.op() == "BiasAddV1";
}

bool IsBiasAddV2(const NodeDef& node) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_23(mht_23_v, 356, "", "./tensorflow/core/grappler/op_types.cc", "IsBiasAddV2");
 return node.op() == "BiasAdd"; }

bool IsBiasAddGrad(const NodeDef& node) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_24(mht_24_v, 361, "", "./tensorflow/core/grappler/op_types.cc", "IsBiasAddGrad");
 return node.op() == "BiasAddGrad"; }

bool IsBitcast(const NodeDef& node) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_25(mht_25_v, 366, "", "./tensorflow/core/grappler/op_types.cc", "IsBitcast");
 return node.op() == "Bitcast"; }

bool IsBroadcastTo(const NodeDef& node) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_26(mht_26_v, 371, "", "./tensorflow/core/grappler/op_types.cc", "IsBroadcastTo");
 return node.op() == "BroadcastTo"; }

bool IsCast(const NodeDef& node) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_27(mht_27_v, 376, "", "./tensorflow/core/grappler/op_types.cc", "IsCast");
 return node.op() == "Cast"; }

bool IsCastLike(const NodeDef& node) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_28(mht_28_v, 381, "", "./tensorflow/core/grappler/op_types.cc", "IsCastLike");

  static const gtl::FlatSet<string>* const kCastLikeOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "Angle", "Bucketize", "Cast", "Dequantize", "HistogramFixedWidth",
          "Imag", "IsFinite", "IsInf", "IsNan", "Quantize",
          "QuantizeDownAndShrinkRange", "QuantizeV2", "QuantizedInstanceNorm",
          "QuantizedRelu", "QuantizedRelu6", "QuantizedReluX", "Real",
          "Requantize"}));
  return kCastLikeOps->count(node.op()) > 0;
}

bool IsCheckNumerics(const NodeDef& node) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_29(mht_29_v, 395, "", "./tensorflow/core/grappler/op_types.cc", "IsCheckNumerics");

  return node.op() == "CheckNumerics";
}

bool IsCollective(const NodeDef& node) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_30(mht_30_v, 402, "", "./tensorflow/core/grappler/op_types.cc", "IsCollective");

  return node.op() == "CollectiveReduce" ||
         node.op() == "CollectiveBcastSend" ||
         node.op() == "CollectiveBcastRecv";
}

bool IsComplex(const NodeDef& node) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_31(mht_31_v, 411, "", "./tensorflow/core/grappler/op_types.cc", "IsComplex");
 return node.op() == "Complex"; }

bool IsComplexAbs(const NodeDef& node) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_32(mht_32_v, 416, "", "./tensorflow/core/grappler/op_types.cc", "IsComplexAbs");
 return node.op() == "ComplexAbs"; }

bool IsConcat(const NodeDef& node) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_33(mht_33_v, 421, "", "./tensorflow/core/grappler/op_types.cc", "IsConcat");

  return node.op() == "Concat" || node.op() == "ConcatV2";
}

bool IsConcatOffset(const NodeDef& node) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_34(mht_34_v, 428, "", "./tensorflow/core/grappler/op_types.cc", "IsConcatOffset");
 return node.op() == "ConcatOffset"; }

bool IsConstant(const NodeDef& node) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_35(mht_35_v, 433, "", "./tensorflow/core/grappler/op_types.cc", "IsConstant");
 return node.op() == "Const"; }

bool IsConj(const NodeDef& node) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_36(mht_36_v, 438, "", "./tensorflow/core/grappler/op_types.cc", "IsConj");
 return node.op() == "Conj"; }

bool IsConjugateTranspose(const NodeDef& node) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_37(mht_37_v, 443, "", "./tensorflow/core/grappler/op_types.cc", "IsConjugateTranspose");

  return node.op() == "ConjugateTranspose";
}

bool IsControlFlow(const NodeDef& node) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_38(mht_38_v, 450, "", "./tensorflow/core/grappler/op_types.cc", "IsControlFlow");

  // clang-format off
  return node.op() == "ControlTrigger" ||
         node.op() == "Enter" ||
         node.op() == "Exit" ||
         node.op() == "LoopCond" ||
         node.op() == "Merge" ||
         node.op() == "_XlaMerge" ||
         node.op() == "NextIteration" ||
         node.op() == "Switch" ||
         node.op() == "_SwitchN";
  // clang-format on
}

bool IsConv2D(const NodeDef& node) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_39(mht_39_v, 467, "", "./tensorflow/core/grappler/op_types.cc", "IsConv2D");
 return node.op() == "Conv2D"; }

bool IsConv2DBackpropFilter(const NodeDef& node) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_40(mht_40_v, 472, "", "./tensorflow/core/grappler/op_types.cc", "IsConv2DBackpropFilter");

  return node.op() == "Conv2DBackpropFilter";
}

bool IsConv2DBackpropInput(const NodeDef& node) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_41(mht_41_v, 479, "", "./tensorflow/core/grappler/op_types.cc", "IsConv2DBackpropInput");

  return node.op() == "Conv2DBackpropInput";
}

bool IsConv3D(const NodeDef& node) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_42(mht_42_v, 486, "", "./tensorflow/core/grappler/op_types.cc", "IsConv3D");
 return node.op() == "Conv3D"; }

bool IsConv3DBackpropFilterV2(const NodeDef& node) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_43(mht_43_v, 491, "", "./tensorflow/core/grappler/op_types.cc", "IsConv3DBackpropFilterV2");

  return node.op() == "Conv3DBackpropFilterV2";
}

bool IsConv3DBackpropInputV2(const NodeDef& node) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_44(mht_44_v, 498, "", "./tensorflow/core/grappler/op_types.cc", "IsConv3DBackpropInputV2");

  return node.op() == "Conv3DBackpropInputV2";
}

bool IsDepthwiseConv2dNative(const NodeDef& node) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_45(mht_45_v, 505, "", "./tensorflow/core/grappler/op_types.cc", "IsDepthwiseConv2dNative");

  return node.op() == "DepthwiseConv2dNative";
}

bool IsDepthwiseConv2dNativeBackpropFilter(const NodeDef& node) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_46(mht_46_v, 512, "", "./tensorflow/core/grappler/op_types.cc", "IsDepthwiseConv2dNativeBackpropFilter");

  return node.op() == "DepthwiseConv2dNativeBackpropFilter";
}

bool IsDepthwiseConv2dNativeBackpropInput(const NodeDef& node) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_47(mht_47_v, 519, "", "./tensorflow/core/grappler/op_types.cc", "IsDepthwiseConv2dNativeBackpropInput");

  return node.op() == "DepthwiseConv2dNativeBackpropInput";
}

bool IsDequeueOp(const NodeDef& node) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_48(mht_48_v, 526, "", "./tensorflow/core/grappler/op_types.cc", "IsDequeueOp");

  const auto& op = node.op();
  return op == "QueueDequeueManyV2" || op == "QueueDequeueMany" ||
         op == "QueueDequeueV2" || op == "QueueDequeue" ||
         op == "QueueDequeueUpToV2" || op == "QueueDequeueUpTo";
}

bool IsDiv(const NodeDef& node) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_49(mht_49_v, 536, "", "./tensorflow/core/grappler/op_types.cc", "IsDiv");
 return node.op() == "Div"; }

bool IsDivNoNan(const NodeDef& node) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_50(mht_50_v, 541, "", "./tensorflow/core/grappler/op_types.cc", "IsDivNoNan");
 return node.op() == "DivNoNan"; }

// Returns true if node represents a unary elementwise function that is
// monotonic. If *is_non_decreasing is true, the function is non-decreasing,
// e.g. sqrt, exp. *is_non_decreasing is false, the function is non-increasing,
// e.g. inv.
bool IsElementWiseMonotonic(const NodeDef& node, bool* is_non_decreasing) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_51(mht_51_v, 550, "", "./tensorflow/core/grappler/op_types.cc", "IsElementWiseMonotonic");

  static const gtl::FlatSet<string>* const kMonotonicNonDecreasingOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "Acosh", "Asin", "Asinh",    "Atan",     "Atanh", "Ceil",
          "Elu",   "Erf",  "Exp",      "Expm1",    "Floor", "Log",
          "Log1p", "Relu", "Relu6",    "Rint",     "Selu",  "Sigmoid",
          "Sign",  "Sinh", "Softsign", "Softplus", "Sqrt",  "Tanh",
      }));
  static const gtl::FlatSet<string>* const kMonotonicNonIncreasingOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{"Acos", "Erfc", "Neg", "Rsqrt"}));
  if (kMonotonicNonDecreasingOps->count(node.op()) > 0) {
    if (is_non_decreasing) {
      *is_non_decreasing = true;
    }
    return true;
  } else if (kMonotonicNonIncreasingOps->count(node.op()) > 0) {
    if (is_non_decreasing) {
      *is_non_decreasing = false;
    }
    return true;
  }
  return false;
}

bool IsElu(const NodeDef& node) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_52(mht_52_v, 577, "", "./tensorflow/core/grappler/op_types.cc", "IsElu");
 return node.op() == "Elu"; }

bool IsEluGrad(const NodeDef& node) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_53(mht_53_v, 582, "", "./tensorflow/core/grappler/op_types.cc", "IsEluGrad");
 return node.op() == "EluGrad"; }

bool IsQuantizationEmulation(const NodeDef& node) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_54(mht_54_v, 587, "", "./tensorflow/core/grappler/op_types.cc", "IsQuantizationEmulation");

  const auto& op = node.op();
  return absl::StartsWith(op, "QuantizeAndDequantize") ||
         absl::StartsWith(op, "FakeQuantWithMinMax");
}

bool IsEnter(const NodeDef& node) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_55(mht_55_v, 596, "", "./tensorflow/core/grappler/op_types.cc", "IsEnter");

  const auto& op = node.op();
  return op == "Enter" || op == "RefEnter";
}

bool IsEqual(const NodeDef& node) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_56(mht_56_v, 604, "", "./tensorflow/core/grappler/op_types.cc", "IsEqual");
 return node.op() == "Equal"; }

bool IsExit(const NodeDef& node) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_57(mht_57_v, 609, "", "./tensorflow/core/grappler/op_types.cc", "IsExit");

  const auto& op = node.op();
  return op == "Exit" || op == "RefExit";
}

bool IsExp(const NodeDef& node) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_58(mht_58_v, 617, "", "./tensorflow/core/grappler/op_types.cc", "IsExp");
 return node.op() == "Exp"; }

bool IsFakeParam(const NodeDef& node) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_59(mht_59_v, 622, "", "./tensorflow/core/grappler/op_types.cc", "IsFakeParam");
 return node.op() == "FakeParam"; }

bool IsFill(const NodeDef& node) {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_60(mht_60_v, 627, "", "./tensorflow/core/grappler/op_types.cc", "IsFill");
 return node.op() == "Fill"; }

bool IsFloorDiv(const NodeDef& node) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_61(mht_61_v, 632, "", "./tensorflow/core/grappler/op_types.cc", "IsFloorDiv");
 return node.op() == "FloorDiv"; }

bool IsFloorMod(const NodeDef& node) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_62(mht_62_v, 637, "", "./tensorflow/core/grappler/op_types.cc", "IsFloorMod");
 return node.op() == "FloorMod"; }

bool IsFusedBatchNorm(const NodeDef& node) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_63(mht_63_v, 642, "", "./tensorflow/core/grappler/op_types.cc", "IsFusedBatchNorm");

  const auto& op = node.op();
  return op == "FusedBatchNorm" || op == "FusedBatchNormV2" ||
         op == "FusedBatchNormV3";
}

bool IsFusedBatchNormEx(const NodeDef& node) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_64(mht_64_v, 651, "", "./tensorflow/core/grappler/op_types.cc", "IsFusedBatchNormEx");

  return node.op() == "_FusedBatchNormEx";
}

bool IsFusedBatchNormGrad(const NodeDef& node) {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_65(mht_65_v, 658, "", "./tensorflow/core/grappler/op_types.cc", "IsFusedBatchNormGrad");

  const auto& op = node.op();
  return op == "FusedBatchNormGrad" || op == "FusedBatchNormGradV2" ||
         op == "FusedBatchNormGradV3";
}

bool IsGather(const NodeDef& node) {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_66(mht_66_v, 667, "", "./tensorflow/core/grappler/op_types.cc", "IsGather");

  const auto& op = node.op();
  return op == "Gather" || op == "GatherV2" || op == "ResourceGather";
}

bool IsGreater(const NodeDef& node) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_67(mht_67_v, 675, "", "./tensorflow/core/grappler/op_types.cc", "IsGreater");
 return node.op() == "Greater"; }

bool IsGreaterEqual(const NodeDef& node) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_68(mht_68_v, 680, "", "./tensorflow/core/grappler/op_types.cc", "IsGreaterEqual");
 return node.op() == "GreaterEqual"; }

bool IsHostConstant(const NodeDef& node) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_69(mht_69_v, 685, "", "./tensorflow/core/grappler/op_types.cc", "IsHostConstant");
 return node.op() == "HostConst"; }

bool IsHistogramSummary(const NodeDef& node) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_70(mht_70_v, 690, "", "./tensorflow/core/grappler/op_types.cc", "IsHistogramSummary");

  return node.op() == "HistogramSummary";
}

bool IsIdentity(const NodeDef& node) {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_71(mht_71_v, 697, "", "./tensorflow/core/grappler/op_types.cc", "IsIdentity");

  const auto& op = node.op();
  return op == "Identity" || op == "RefIdentity";
}

bool IsIdentityN(const NodeDef& node) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_72(mht_72_v, 705, "", "./tensorflow/core/grappler/op_types.cc", "IsIdentityN");

  const auto& op = node.op();
  return op == "IdentityN";
}

bool IsIdentityNSingleInput(const NodeDef& node) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_73(mht_73_v, 713, "", "./tensorflow/core/grappler/op_types.cc", "IsIdentityNSingleInput");

  return IsIdentityN(node) && node.attr().count("T") != 0 &&
         node.attr().at("T").list().type_size() == 1;
}

bool IsIf(const NodeDef& node) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_74(mht_74_v, 721, "", "./tensorflow/core/grappler/op_types.cc", "IsIf");

  const auto& op = node.op();
  return op == "If" || op == "StatelessIf";
}

bool IsIgamma(const NodeDef& node) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_75(mht_75_v, 729, "", "./tensorflow/core/grappler/op_types.cc", "IsIgamma");
 return node.op() == "Igamma"; }

bool IsIgammac(const NodeDef& node) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_76(mht_76_v, 734, "", "./tensorflow/core/grappler/op_types.cc", "IsIgammac");
 return node.op() == "Igammac"; }

bool IsImag(const NodeDef& node) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_77(mht_77_v, 739, "", "./tensorflow/core/grappler/op_types.cc", "IsImag");
 return node.op() == "Imag"; }

bool IsImmutableConst(const NodeDef& node) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_78(mht_78_v, 744, "", "./tensorflow/core/grappler/op_types.cc", "IsImmutableConst");

  return node.op() == "ImmutableConst";
}

bool IsInvGrad(const NodeDef& node) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_79(mht_79_v, 751, "", "./tensorflow/core/grappler/op_types.cc", "IsInvGrad");
 return node.op() == "InvGrad"; }

bool IsLeakyRelu(const NodeDef& node) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_80(mht_80_v, 756, "", "./tensorflow/core/grappler/op_types.cc", "IsLeakyRelu");
 return node.op() == "LeakyRelu"; }

bool IsLeakyReluGrad(const NodeDef& node) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_81(mht_81_v, 761, "", "./tensorflow/core/grappler/op_types.cc", "IsLeakyReluGrad");

  return node.op() == "LeakyReluGrad";
}

bool IsLess(const NodeDef& node) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_82(mht_82_v, 768, "", "./tensorflow/core/grappler/op_types.cc", "IsLess");
 return node.op() == "Less"; }

bool IsLessEqual(const NodeDef& node) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_83(mht_83_v, 773, "", "./tensorflow/core/grappler/op_types.cc", "IsLessEqual");
 return node.op() == "LessEqual"; }

bool IsLog(const NodeDef& node) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_84(mht_84_v, 778, "", "./tensorflow/core/grappler/op_types.cc", "IsLog");
 return node.op() == "Log"; }

bool IsLogicalAnd(const NodeDef& node) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_85(mht_85_v, 783, "", "./tensorflow/core/grappler/op_types.cc", "IsLogicalAnd");
 return node.op() == "LogicalAnd"; }

bool IsLogicalNot(const NodeDef& node) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_86(mht_86_v, 788, "", "./tensorflow/core/grappler/op_types.cc", "IsLogicalNot");
 return node.op() == "LogicalNot"; }

bool IsLogicalOr(const NodeDef& node) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_87(mht_87_v, 793, "", "./tensorflow/core/grappler/op_types.cc", "IsLogicalOr");
 return node.op() == "LogicalOr"; }

bool IsLoopCond(const NodeDef& node) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_88(mht_88_v, 798, "", "./tensorflow/core/grappler/op_types.cc", "IsLoopCond");
 return node.op() == "LoopCond"; }

bool IsMatMul(const NodeDef& node) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_89(mht_89_v, 803, "", "./tensorflow/core/grappler/op_types.cc", "IsMatMul");
 return node.op() == "MatMul"; }

bool IsMax(const NodeDef& node) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_90(mht_90_v, 808, "", "./tensorflow/core/grappler/op_types.cc", "IsMax");
 return node.op() == "Max"; }

bool IsMaximum(const NodeDef& node) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_91(mht_91_v, 813, "", "./tensorflow/core/grappler/op_types.cc", "IsMaximum");
 return node.op() == "Maximum"; }

bool IsMaxPoolGrad(const NodeDef& node) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_92(mht_92_v, 818, "", "./tensorflow/core/grappler/op_types.cc", "IsMaxPoolGrad");
 return node.op() == "MaxPoolGrad"; }

bool IsMean(const NodeDef& node) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_93(mht_93_v, 823, "", "./tensorflow/core/grappler/op_types.cc", "IsMean");
 return node.op() == "Mean"; }

bool IsMerge(const NodeDef& node) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_94(mht_94_v, 828, "", "./tensorflow/core/grappler/op_types.cc", "IsMerge");

  const auto& op = node.op();
  return op == "Merge" || op == "RefMerge" || op == "_XlaMerge";
}

bool IsMin(const NodeDef& node) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_95(mht_95_v, 836, "", "./tensorflow/core/grappler/op_types.cc", "IsMin");
 return node.op() == "Min"; }

bool IsMinimum(const NodeDef& node) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_96(mht_96_v, 841, "", "./tensorflow/core/grappler/op_types.cc", "IsMinimum");
 return node.op() == "Minimum"; }

bool IsMirrorPad(const NodeDef& node) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_97(mht_97_v, 846, "", "./tensorflow/core/grappler/op_types.cc", "IsMirrorPad");
 return node.op() == "MirrorPad"; }

bool IsMirrorPadGrad(const NodeDef& node) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_98(mht_98_v, 851, "", "./tensorflow/core/grappler/op_types.cc", "IsMirrorPadGrad");

  return node.op() == "MirrorPadGrad";
}

bool IsMod(const NodeDef& node) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_99(mht_99_v, 858, "", "./tensorflow/core/grappler/op_types.cc", "IsMod");
 return node.op() == "Mod"; }

bool IsMul(const NodeDef& node) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_100(mht_100_v, 863, "", "./tensorflow/core/grappler/op_types.cc", "IsMul");
 return node.op() == "Mul"; }
bool IsMulNoNan(const NodeDef& node) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_101(mht_101_v, 867, "", "./tensorflow/core/grappler/op_types.cc", "IsMulNoNan");
 return node.op() == "MulNoNan"; }
bool IsAnyMul(const NodeDef& node) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_102(mht_102_v, 871, "", "./tensorflow/core/grappler/op_types.cc", "IsAnyMul");
 return IsMul(node) || IsMulNoNan(node); }

bool IsNeg(const NodeDef& node) {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_103(mht_103_v, 876, "", "./tensorflow/core/grappler/op_types.cc", "IsNeg");
 return node.op() == "Neg"; }

bool IsNoOp(const NodeDef& node) {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_104(mht_104_v, 881, "", "./tensorflow/core/grappler/op_types.cc", "IsNoOp");
 return node.op() == "NoOp"; }

bool IsNotEqual(const NodeDef& node) {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_105(mht_105_v, 886, "", "./tensorflow/core/grappler/op_types.cc", "IsNotEqual");
 return node.op() == "NotEqual"; }

bool IsNextIteration(const NodeDef& node) {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_106(mht_106_v, 891, "", "./tensorflow/core/grappler/op_types.cc", "IsNextIteration");

  const auto& op = node.op();
  return op == "NextIteration" || op == "RefNextIteration";
}

bool IsOnesLike(const NodeDef& node) {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_107(mht_107_v, 899, "", "./tensorflow/core/grappler/op_types.cc", "IsOnesLike");
 return node.op() == "OnesLike"; }

bool IsPack(const NodeDef& node) {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_108(mht_108_v, 904, "", "./tensorflow/core/grappler/op_types.cc", "IsPack");
 return node.op() == "Pack"; }

bool IsPad(const NodeDef& node) {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_109(mht_109_v, 909, "", "./tensorflow/core/grappler/op_types.cc", "IsPad");

  const auto& op = node.op();
  return op == "Pad" || op == "PadV2";
}

bool IsPartitionedCall(const NodeDef& node) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_110(mht_110_v, 917, "", "./tensorflow/core/grappler/op_types.cc", "IsPartitionedCall");

  return node.op() == "PartitionedCall";
}

bool IsPlaceholder(const NodeDef& node) {
   std::vector<std::string> mht_111_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_111(mht_111_v, 924, "", "./tensorflow/core/grappler/op_types.cc", "IsPlaceholder");

  const auto& op = node.op();
  return op == "Placeholder" || op == "PlaceholderV2" ||
         op == "PlaceholderWithDefault";
}

bool IsPolygamma(const NodeDef& node) {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_112(mht_112_v, 933, "", "./tensorflow/core/grappler/op_types.cc", "IsPolygamma");
 return node.op() == "Polygamma"; }

bool IsPow(const NodeDef& node) {
   std::vector<std::string> mht_113_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_113(mht_113_v, 938, "", "./tensorflow/core/grappler/op_types.cc", "IsPow");
 return node.op() == "Pow"; }

bool IsPrint(const NodeDef& node) {
   std::vector<std::string> mht_114_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_114(mht_114_v, 943, "", "./tensorflow/core/grappler/op_types.cc", "IsPrint");

  return node.op() == "Print" || node.op() == "PrintV2";
}

bool IsProd(const NodeDef& node) {
   std::vector<std::string> mht_115_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_115(mht_115_v, 950, "", "./tensorflow/core/grappler/op_types.cc", "IsProd");
 return node.op() == "Prod"; }

bool IsQuantizedMatMul(const NodeDef& node) {
   std::vector<std::string> mht_116_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_116(mht_116_v, 955, "", "./tensorflow/core/grappler/op_types.cc", "IsQuantizedMatMul");

  return node.op() == "QuantizedMatMul" || node.op() == "QuantizedMatMulV2";
}

bool IsQueue(const NodeDef& node) {
   std::vector<std::string> mht_117_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_117(mht_117_v, 962, "", "./tensorflow/core/grappler/op_types.cc", "IsQueue");

  return str_util::EndsWith(node.op(), "QueueV2");
}

bool IsRandomShuffle(const NodeDef& node) {
   std::vector<std::string> mht_118_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_118(mht_118_v, 969, "", "./tensorflow/core/grappler/op_types.cc", "IsRandomShuffle");

  return node.op() == "RandomShuffle";
}

bool IsRank(const NodeDef& node) {
   std::vector<std::string> mht_119_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_119(mht_119_v, 976, "", "./tensorflow/core/grappler/op_types.cc", "IsRank");
 return node.op() == "Rank"; }

bool IsReadVariableOp(const NodeDef& node) {
   std::vector<std::string> mht_120_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_120(mht_120_v, 981, "", "./tensorflow/core/grappler/op_types.cc", "IsReadVariableOp");

  return node.op() == "ReadVariableOp";
}

bool IsReadVariablesOp(const NodeDef& node) {
   std::vector<std::string> mht_121_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_121(mht_121_v, 988, "", "./tensorflow/core/grappler/op_types.cc", "IsReadVariablesOp");

  return node.op() == "_ReadVariablesOp";
}

bool IsReal(const NodeDef& node) {
   std::vector<std::string> mht_122_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_122(mht_122_v, 995, "", "./tensorflow/core/grappler/op_types.cc", "IsReal");
 return node.op() == "Real"; }

bool IsRealDiv(const NodeDef& node) {
   std::vector<std::string> mht_123_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_123(mht_123_v, 1000, "", "./tensorflow/core/grappler/op_types.cc", "IsRealDiv");
 return node.op() == "RealDiv"; }

bool IsReciprocalGrad(const NodeDef& node) {
   std::vector<std::string> mht_124_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_124(mht_124_v, 1005, "", "./tensorflow/core/grappler/op_types.cc", "IsReciprocalGrad");

  return node.op() == "ReciprocalGrad";
}

bool IsRecv(const NodeDef& node) {
   std::vector<std::string> mht_125_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_125(mht_125_v, 1012, "", "./tensorflow/core/grappler/op_types.cc", "IsRecv");

  return node.op() == "_Recv" || node.op() == "_HostRecv";
}

bool IsReduction(const NodeDef& node) {
   std::vector<std::string> mht_126_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_126(mht_126_v, 1019, "", "./tensorflow/core/grappler/op_types.cc", "IsReduction");

  const auto& op = node.op();
  return op == "Sum" || op == "Prod" || op == "Min" || op == "Max" ||
         op == "Mean" || op == "Any" || op == "All";
}

bool IsRelu(const NodeDef& node) {
   std::vector<std::string> mht_127_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_127(mht_127_v, 1028, "", "./tensorflow/core/grappler/op_types.cc", "IsRelu");
 return node.op() == "Relu"; }

bool IsRelu6(const NodeDef& node) {
   std::vector<std::string> mht_128_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_128(mht_128_v, 1033, "", "./tensorflow/core/grappler/op_types.cc", "IsRelu6");
 return node.op() == "Relu6"; }

bool IsReluGrad(const NodeDef& node) {
   std::vector<std::string> mht_129_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_129(mht_129_v, 1038, "", "./tensorflow/core/grappler/op_types.cc", "IsReluGrad");
 return node.op() == "ReluGrad"; }

bool IsRelu6Grad(const NodeDef& node) {
   std::vector<std::string> mht_130_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_130(mht_130_v, 1043, "", "./tensorflow/core/grappler/op_types.cc", "IsRelu6Grad");
 return node.op() == "Relu6Grad"; }

bool IsReshape(const NodeDef& node) {
   std::vector<std::string> mht_131_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_131(mht_131_v, 1048, "", "./tensorflow/core/grappler/op_types.cc", "IsReshape");
 return (node.op() == "Reshape"); }

bool IsRestore(const NodeDef& node) {
   std::vector<std::string> mht_132_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_132(mht_132_v, 1053, "", "./tensorflow/core/grappler/op_types.cc", "IsRestore");

  return (node.op() == "Restore" || node.op() == "RestoreV2" ||
          node.op() == "RestoreSlice");
}

bool IsRetval(const NodeDef& node) {
   std::vector<std::string> mht_133_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_133(mht_133_v, 1061, "", "./tensorflow/core/grappler/op_types.cc", "IsRetval");

  return node.op() == "_Retval" || node.op() == "_DeviceRetval";
}

bool IsReverse(const NodeDef& node) {
   std::vector<std::string> mht_134_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_134(mht_134_v, 1068, "", "./tensorflow/core/grappler/op_types.cc", "IsReverse");

  return node.op() == "Reverse" || node.op() == "ReverseV2";
}

bool IsReverseV2(const NodeDef& node) {
   std::vector<std::string> mht_135_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_135(mht_135_v, 1075, "", "./tensorflow/core/grappler/op_types.cc", "IsReverseV2");
 return node.op() == "ReverseV2"; }

bool IsRsqrt(const NodeDef& node) {
   std::vector<std::string> mht_136_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_136(mht_136_v, 1080, "", "./tensorflow/core/grappler/op_types.cc", "IsRsqrt");
 return node.op() == "Rsqrt"; }

bool IsRsqrtGrad(const NodeDef& node) {
   std::vector<std::string> mht_137_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_137(mht_137_v, 1085, "", "./tensorflow/core/grappler/op_types.cc", "IsRsqrtGrad");
 return node.op() == "RsqrtGrad"; }

bool IsSelect(const NodeDef& node) {
   std::vector<std::string> mht_138_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_138(mht_138_v, 1090, "", "./tensorflow/core/grappler/op_types.cc", "IsSelect");

  return node.op() == "Select" || node.op() == "SelectV2";
}

bool IsSeluGrad(const NodeDef& node) {
   std::vector<std::string> mht_139_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_139(mht_139_v, 1097, "", "./tensorflow/core/grappler/op_types.cc", "IsSeluGrad");
 return node.op() == "SeluGrad"; }

bool IsSend(const NodeDef& node) {
   std::vector<std::string> mht_140_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_140(mht_140_v, 1102, "", "./tensorflow/core/grappler/op_types.cc", "IsSend");

  return node.op() == "_Send" || node.op() == "_HostSend";
}

bool IsShape(const NodeDef& node) {
   std::vector<std::string> mht_141_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_141(mht_141_v, 1109, "", "./tensorflow/core/grappler/op_types.cc", "IsShape");
 return node.op() == "Shape"; }

bool IsShapeN(const NodeDef& node) {
   std::vector<std::string> mht_142_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_142(mht_142_v, 1114, "", "./tensorflow/core/grappler/op_types.cc", "IsShapeN");
 return node.op() == "ShapeN"; }

bool IsShuffle(const NodeDef& node) {
   std::vector<std::string> mht_143_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_143(mht_143_v, 1119, "", "./tensorflow/core/grappler/op_types.cc", "IsShuffle");
 return node.op() == "Shuffle"; }

bool IsSigmoid(const NodeDef& node) {
   std::vector<std::string> mht_144_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_144(mht_144_v, 1124, "", "./tensorflow/core/grappler/op_types.cc", "IsSigmoid");
 return node.op() == "Sigmoid"; }

bool IsSigmoidGrad(const NodeDef& node) {
   std::vector<std::string> mht_145_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_145(mht_145_v, 1129, "", "./tensorflow/core/grappler/op_types.cc", "IsSigmoidGrad");
 return node.op() == "SigmoidGrad"; }

bool IsSize(const NodeDef& node) {
   std::vector<std::string> mht_146_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_146(mht_146_v, 1134, "", "./tensorflow/core/grappler/op_types.cc", "IsSize");
 return node.op() == "Size"; }

bool IsSlice(const NodeDef& node) {
   std::vector<std::string> mht_147_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_147(mht_147_v, 1139, "", "./tensorflow/core/grappler/op_types.cc", "IsSlice");
 return node.op() == "Slice"; }

bool IsSnapshot(const NodeDef& node) {
   std::vector<std::string> mht_148_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_148(mht_148_v, 1144, "", "./tensorflow/core/grappler/op_types.cc", "IsSnapshot");
 return node.op() == "Snapshot"; }

bool IsSoftmax(const NodeDef& node) {
   std::vector<std::string> mht_149_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_149(mht_149_v, 1149, "", "./tensorflow/core/grappler/op_types.cc", "IsSoftmax");
 return node.op() == "Softmax"; }

bool IsSoftplusGrad(const NodeDef& node) {
   std::vector<std::string> mht_150_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_150(mht_150_v, 1154, "", "./tensorflow/core/grappler/op_types.cc", "IsSoftplusGrad");
 return node.op() == "SoftplusGrad"; }

bool IsSoftsignGrad(const NodeDef& node) {
   std::vector<std::string> mht_151_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_151(mht_151_v, 1159, "", "./tensorflow/core/grappler/op_types.cc", "IsSoftsignGrad");
 return node.op() == "SoftsignGrad"; }

bool IsSplit(const NodeDef& node) {
   std::vector<std::string> mht_152_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_152(mht_152_v, 1164, "", "./tensorflow/core/grappler/op_types.cc", "IsSplit");
 return node.op() == "Split"; }

bool IsSplitV(const NodeDef& node) {
   std::vector<std::string> mht_153_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_153(mht_153_v, 1169, "", "./tensorflow/core/grappler/op_types.cc", "IsSplitV");
 return node.op() == "SplitV"; }

bool IsSqrt(const NodeDef& node) {
   std::vector<std::string> mht_154_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_154(mht_154_v, 1174, "", "./tensorflow/core/grappler/op_types.cc", "IsSqrt");
 return node.op() == "Sqrt"; }

bool IsSqrtGrad(const NodeDef& node) {
   std::vector<std::string> mht_155_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_155(mht_155_v, 1179, "", "./tensorflow/core/grappler/op_types.cc", "IsSqrtGrad");
 return node.op() == "SqrtGrad"; }

bool IsSquare(const NodeDef& node) {
   std::vector<std::string> mht_156_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_156(mht_156_v, 1184, "", "./tensorflow/core/grappler/op_types.cc", "IsSquare");
 return node.op() == "Square"; }

bool IsSquaredDifference(const NodeDef& node) {
   std::vector<std::string> mht_157_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_157(mht_157_v, 1189, "", "./tensorflow/core/grappler/op_types.cc", "IsSquaredDifference");

  return node.op() == "SquaredDifference";
}

bool IsSqueeze(const NodeDef& node) {
   std::vector<std::string> mht_158_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_158(mht_158_v, 1196, "", "./tensorflow/core/grappler/op_types.cc", "IsSqueeze");
 return node.op() == "Squeeze"; }

bool IsStackOp(const NodeDef& node) {
   std::vector<std::string> mht_159_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_159(mht_159_v, 1201, "", "./tensorflow/core/grappler/op_types.cc", "IsStackOp");

  return node.op() == "Stack" || node.op() == "StackV2";
}
bool IsStackCloseOp(const NodeDef& node) {
   std::vector<std::string> mht_160_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_160(mht_160_v, 1207, "", "./tensorflow/core/grappler/op_types.cc", "IsStackCloseOp");

  return node.op() == "StackClose" || node.op() == "StackCloseV2";
}
bool IsStackPushOp(const NodeDef& node) {
   std::vector<std::string> mht_161_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_161(mht_161_v, 1213, "", "./tensorflow/core/grappler/op_types.cc", "IsStackPushOp");

  return node.op() == "StackPush" || node.op() == "StackPushV2";
}
bool IsStackPopOp(const NodeDef& node) {
   std::vector<std::string> mht_162_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_162(mht_162_v, 1219, "", "./tensorflow/core/grappler/op_types.cc", "IsStackPopOp");

  return node.op() == "StackPop" || node.op() == "StackPopV2";
}

bool IsStatefulPartitionedCall(const NodeDef& node) {
   std::vector<std::string> mht_163_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_163(mht_163_v, 1226, "", "./tensorflow/core/grappler/op_types.cc", "IsStatefulPartitionedCall");

  return node.op() == "StatefulPartitionedCall";
}

bool IsStopGradient(const NodeDef& node) {
   std::vector<std::string> mht_164_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_164(mht_164_v, 1233, "", "./tensorflow/core/grappler/op_types.cc", "IsStopGradient");

  const auto& op = node.op();
  return op == "StopGradient" || op == "PreventGradient";
}

bool IsStridedSlice(const NodeDef& node) {
   std::vector<std::string> mht_165_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_165(mht_165_v, 1241, "", "./tensorflow/core/grappler/op_types.cc", "IsStridedSlice");
 return node.op() == "StridedSlice"; }

bool IsStridedSliceGrad(const NodeDef& node) {
   std::vector<std::string> mht_166_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_166(mht_166_v, 1246, "", "./tensorflow/core/grappler/op_types.cc", "IsStridedSliceGrad");

  return node.op() == "StridedSliceGrad";
}

bool IsStringToHashBucketFast(const NodeDef& node) {
   std::vector<std::string> mht_167_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_167(mht_167_v, 1253, "", "./tensorflow/core/grappler/op_types.cc", "IsStringToHashBucketFast");

  return node.op() == "StringToHashBucketFast";
}

bool IsSub(const NodeDef& node) {
   std::vector<std::string> mht_168_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_168(mht_168_v, 1260, "", "./tensorflow/core/grappler/op_types.cc", "IsSub");
 return node.op() == "Sub"; }

bool IsSum(const NodeDef& node) {
   std::vector<std::string> mht_169_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_169(mht_169_v, 1265, "", "./tensorflow/core/grappler/op_types.cc", "IsSum");
 return node.op() == "Sum"; }

bool IsSwitch(const NodeDef& node) {
   std::vector<std::string> mht_170_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_170(mht_170_v, 1270, "", "./tensorflow/core/grappler/op_types.cc", "IsSwitch");

  const auto& op = node.op();
  return op == "_SwitchN" || op == "Switch" || op == "RefSwitch";
}

bool IsSymbolicGradient(const NodeDef& node) {
   std::vector<std::string> mht_171_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_171(mht_171_v, 1278, "", "./tensorflow/core/grappler/op_types.cc", "IsSymbolicGradient");

  return node.op() == "SymbolicGradient";
}

bool IsTanh(const NodeDef& node) {
   std::vector<std::string> mht_172_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_172(mht_172_v, 1285, "", "./tensorflow/core/grappler/op_types.cc", "IsTanh");
 return node.op() == "Tanh"; }

bool IsTanhGrad(const NodeDef& node) {
   std::vector<std::string> mht_173_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_173(mht_173_v, 1290, "", "./tensorflow/core/grappler/op_types.cc", "IsTanhGrad");
 return node.op() == "TanhGrad"; }

bool IsTensorArray(const NodeDef& node) {
   std::vector<std::string> mht_174_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_174(mht_174_v, 1295, "", "./tensorflow/core/grappler/op_types.cc", "IsTensorArray");

  static const gtl::FlatSet<string>* const kTensorArrayOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "TensorArray",
          "TensorArrayV2",
          "TensorArrayV3",
          "TensorArrayGrad",
          "TensorArrayGradV2",
          "TensorArrayGradV3",
          "TensorArrayGradWithShape",
          "TensorArrayWrite",
          "TensorArrayWriteV2",
          "TensorArrayWriteV3",
          "TensorArrayRead",
          "TensorArrayReadV2",
          "TensorArrayReadV3",
          "TensorArrayConcat",
          "TensorArrayConcatV2",
          "TensorArrayConcatV3",
          "TensorArraySplit",
          "TensorArraySplitV2",
          "TensorArraySplitV3",
          "TensorArraySize",
          "TensorArraySizeV2",
          "TensorArraySizeV3",
          "TensorArrayClose",
          "TensorArrayCloseV2",
          "TensorArrayCloseV3",
      }));
  return kTensorArrayOps->count(node.op()) > 0;
}

bool IsTile(const NodeDef& node) {
   std::vector<std::string> mht_175_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_175(mht_175_v, 1330, "", "./tensorflow/core/grappler/op_types.cc", "IsTile");
 return node.op() == "Tile"; }

bool IsTranspose(const NodeDef& node) {
   std::vector<std::string> mht_176_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_176(mht_176_v, 1335, "", "./tensorflow/core/grappler/op_types.cc", "IsTranspose");
 return node.op() == "Transpose"; }

bool IsTruncateDiv(const NodeDef& node) {
   std::vector<std::string> mht_177_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_177(mht_177_v, 1340, "", "./tensorflow/core/grappler/op_types.cc", "IsTruncateDiv");
 return node.op() == "TruncateDiv"; }

bool IsTruncateMod(const NodeDef& node) {
   std::vector<std::string> mht_178_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_178(mht_178_v, 1345, "", "./tensorflow/core/grappler/op_types.cc", "IsTruncateMod");
 return node.op() == "TruncateMod"; }

bool IsUnique(const NodeDef& node) {
   std::vector<std::string> mht_179_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_179(mht_179_v, 1350, "", "./tensorflow/core/grappler/op_types.cc", "IsUnique");

  const auto& op = node.op();
  return op == "Unique" || op == "UniqueV2";
}

bool IsUnpack(const NodeDef& node) {
   std::vector<std::string> mht_180_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_180(mht_180_v, 1358, "", "./tensorflow/core/grappler/op_types.cc", "IsUnpack");
 return node.op() == "Unpack"; }

bool IsVariable(const NodeDef& node) {
   std::vector<std::string> mht_181_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_181(mht_181_v, 1363, "", "./tensorflow/core/grappler/op_types.cc", "IsVariable");

  const auto& op = node.op();
  return op == "Variable" || op == "VariableV2" || op == "AutoReloadVariable" ||
         op == "VarHandleOp" || op == "ReadVariableOp" ||
         op == "_VarHandlesOp" || op == "_ReadVariablesOp";
}

bool IsWhile(const NodeDef& node) {
   std::vector<std::string> mht_182_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_182(mht_182_v, 1373, "", "./tensorflow/core/grappler/op_types.cc", "IsWhile");

  const auto& op = node.op();
  return op == "While" || op == "StatelessWhile";
}

bool IsXdivy(const NodeDef& node) {
   std::vector<std::string> mht_183_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_183(mht_183_v, 1381, "", "./tensorflow/core/grappler/op_types.cc", "IsXdivy");
 return node.op() == "Xdivy"; }

bool IsZerosLike(const NodeDef& node) {
   std::vector<std::string> mht_184_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_184(mht_184_v, 1386, "", "./tensorflow/core/grappler/op_types.cc", "IsZerosLike");
 return node.op() == "ZerosLike"; }

bool IsZeta(const NodeDef& node) {
   std::vector<std::string> mht_185_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_185(mht_185_v, 1391, "", "./tensorflow/core/grappler/op_types.cc", "IsZeta");
 return node.op() == "Zeta"; }

namespace {
bool GetBoolAttr(const NodeDef& node, const string& name) {
   std::vector<std::string> mht_186_v;
   mht_186_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_186(mht_186_v, 1398, "", "./tensorflow/core/grappler/op_types.cc", "GetBoolAttr");

  return node.attr().count(name) > 0 && node.attr().at(name).b();
}
}  // namespace

bool IsPersistent(const NodeDef& node) {
   std::vector<std::string> mht_187_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_187(mht_187_v, 1406, "", "./tensorflow/core/grappler/op_types.cc", "IsPersistent");

  return IsConstant(node) || IsVariable(node) || IsHostConstant(node);
}

bool HasRefInput(const NodeDef& node) {
   std::vector<std::string> mht_188_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_188(mht_188_v, 1413, "", "./tensorflow/core/grappler/op_types.cc", "HasRefInput");

  const OpDef* op_def;
  Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (!status.ok()) {
    return false;
  }
  // Nodes such as Assign or AssignAdd modify one of their inputs.
  for (const auto& input : op_def->input_arg()) {
    if (input.is_ref()) {
      return true;
    }
  }
  return false;
}

bool IsDataset(const NodeDef& node) {
   std::vector<std::string> mht_189_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_189(mht_189_v, 1431, "", "./tensorflow/core/grappler/op_types.cc", "IsDataset");

  const string& op = node.op();
  // See `GetNodeClassForOp` in core/graph/graph.cc.
  return op == "IteratorGetNext" || op == "IteratorGetNextSync" ||
         op == "DatasetToSingleElement" || op == "ReduceDataset";
}

bool IsStateful(const NodeDef node, const OpRegistryInterface* op_registry) {
   std::vector<std::string> mht_190_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_190(mht_190_v, 1441, "", "./tensorflow/core/grappler/op_types.cc", "IsStateful");

  const OpDef* op_def = nullptr;
  const string& op_name = node.op();
  Status status = op_registry->LookUpOpDef(op_name, &op_def);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to lookup OpDef for " << op_name
                 << ". Error: " << status.error_message();
    return false;
  }
  return op_def->is_stateful();
}

bool IsStateful(const NodeDef node) {
   std::vector<std::string> mht_191_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_191(mht_191_v, 1456, "", "./tensorflow/core/grappler/op_types.cc", "IsStateful");

  return IsStateful(node, OpRegistry::Global());
}

bool IsFreeOfSideEffect(const NodeDef& node,
                        const OpRegistryInterface* op_registry) {
   std::vector<std::string> mht_192_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_192(mht_192_v, 1464, "", "./tensorflow/core/grappler/op_types.cc", "IsFreeOfSideEffect");

  // Placeholders must be preserved to keep the graph feedable.
  if (IsPlaceholder(node)) {
    return false;
  }
  const OpDef* op_def = nullptr;
  const string& op_name = node.op();
  Status status = op_registry->LookUpOpDef(op_name, &op_def);
  if (!status.ok()) {
    return false;
  }
  if (op_def->is_stateful()) {
    return false;
  }
  // Nodes such as Assign or AssignAdd modify one of their inputs.
  for (const auto& input : op_def->input_arg()) {
    if (input.is_ref()) {
      return false;
    }
  }
  // Queue ops modify the queue which is a side effect.
  if (node.op().find("Queue") != string::npos) {
    return false;
  }
  // Sending a tensor via a network is a side effect.
  if (IsSend(node)) {
    return false;
  }
  return !ModifiesInputsInPlace(node);
}

bool IsFreeOfSideEffect(const NodeDef& node) {
   std::vector<std::string> mht_193_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_193(mht_193_v, 1498, "", "./tensorflow/core/grappler/op_types.cc", "IsFreeOfSideEffect");

  return IsFreeOfSideEffect(node, OpRegistry::Global());
}

bool ModifiesInputsInPlace(const NodeDef& node) {
   std::vector<std::string> mht_194_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_194(mht_194_v, 1505, "", "./tensorflow/core/grappler/op_types.cc", "ModifiesInputsInPlace");

  // Some nodes do in-place updates on regular tensor inputs.
  const string& op_name = node.op();

  // Ops that modify resource variables effectively modify one of their inputs.
  if (op_name == "AssignVariableOp" || op_name == "AssignAddVariableOp" ||
      op_name == "AssignSubVariableOp" || op_name == "ResourceScatterUpdate" ||
      op_name == "ResourceScatterAdd" || op_name == "ResourceScatterSub" ||
      op_name == "ResourceScatterMul" || op_name == "ResourceScatterDiv" ||
      op_name == "ResourceScatterMin" || op_name == "ResourceScatterMax") {
    return false;
  }

  string lower_op_name = op_name;
  std::transform(lower_op_name.begin(), lower_op_name.end(),
                 lower_op_name.begin(), ::tolower);
  if (absl::StrContains(lower_op_name, "inplace")) {
    return true;
  }
  return GetBoolAttr(node, "in_place") || GetBoolAttr(node, "inplace");
}

bool ModifiesFrameInfo(const NodeDef& node) {
   std::vector<std::string> mht_195_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_195(mht_195_v, 1530, "", "./tensorflow/core/grappler/op_types.cc", "ModifiesFrameInfo");

  return IsEnter(node) || IsExit(node) || IsNextIteration(node);
}

#define OPDEF_PROPERTY_HELPER(PROPERTY_CAP, PROPERTY)                      \
  bool Is##PROPERTY_CAP(const NodeDef& node) {                             \
    if (node.op() == "Add") {                                              \
      /* Workaround for "Add" not being marked is_commutative and */       \
      /* is_aggregate. (See cl/173915048). */                              \
      const auto type = GetDataTypeFromAttr(node, "T");                    \
      return type != DT_INVALID && type != DT_STRING;                      \
    }                                                                      \
    const OpDef* op_def = nullptr;                                         \
    Status status = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def); \
    return status.ok() && op_def->is_##PROPERTY();                         \
  }

OPDEF_PROPERTY_HELPER(Aggregate, aggregate)
OPDEF_PROPERTY_HELPER(Commutative, commutative)

bool IsInvolution(const NodeDef& node) {
  static const gtl::FlatSet<string>* const kInvolutionOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{"Conj", "Reciprocal", "Invert",
                                              "Neg", "LogicalNot"}));
  return kInvolutionOps->count(node.op()) > 0;
}

bool IsValueAndOrderAndShapePreserving(const NodeDef& node) {
   std::vector<std::string> mht_196_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_196(mht_196_v, 1560, "", "./tensorflow/core/grappler/op_types.cc", "IsValueAndOrderAndShapePreserving");

  if (NumNonControlInputs(node) == 1 && IsAggregate(node)) {
    return true;
  }
  static const gtl::FlatSet<string>* const kValueAndOrderAndShapePreservingOps =
      CHECK_NOTNULL((new const gtl::FlatSet<string>{
          "CheckNumerics",
          "DebugGradientIdentity",
          "DeepCopy",
          "Enter",
          "Exit",
          "PreventGradient",
          "Print",
          "Snapshot",
          "StopGradient",
      }));
  return kValueAndOrderAndShapePreservingOps->count(node.op()) > 0 ||
         IsIdentity(node);
}

bool IsValueAndOrderPreserving(const NodeDef& node) {
   std::vector<std::string> mht_197_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_197(mht_197_v, 1583, "", "./tensorflow/core/grappler/op_types.cc", "IsValueAndOrderPreserving");

  if (NumNonControlInputs(node) == 1 && IsAggregate(node)) {
    return true;
  }
  static const gtl::FlatSet<string>* const kValueAndOrderPreservingOps =
      CHECK_NOTNULL((new const gtl::FlatSet<string>{
          "ExpandDims",
          "Reshape",
          "Squeeze",
      }));
  return kValueAndOrderPreservingOps->count(node.op()) > 0 ||
         IsValueAndOrderAndShapePreserving(node);
}

bool IsValuePreserving(const NodeDef& node) {
   std::vector<std::string> mht_198_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_198(mht_198_v, 1600, "", "./tensorflow/core/grappler/op_types.cc", "IsValuePreserving");

  static const gtl::FlatSet<string>* const kValuePreservingOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "InvertPermutation",
          "Reverse",
          "ReverseV2",
          "Roll",
          "Transpose",
          "DepthToSpace",
          "SpaceToDepth",
          "BatchToSpace",
          "BatchToSpaceND",
          "SpaceToBatch",
          "SpaceToBatchND",
      }));
  return IsValueAndOrderPreserving(node) ||
         kValuePreservingOps->count(node.op()) > 0;
}

bool IsUnaryElementWise(const NodeDef& node) {
   std::vector<std::string> mht_199_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_199(mht_199_v, 1622, "", "./tensorflow/core/grappler/op_types.cc", "IsUnaryElementWise");

  static const gtl::FlatSet<string>* const kElementWiseOps =
      CHECK_NOTNULL((new gtl::FlatSet<string>{
          "Abs",      "Acos",     "Acosh",      "Asin",       "Asinh",
          "Atan",     "Atanh",    "Ceil",       "ComplexAbs", "Conj",
          "Cos",      "Cosh",     "Digamma",    "Elu",        "Erf",
          "Erfc",     "Exp",      "Expm1",      "Floor",      "Inv",
          "Invert",   "Isinf",    "Isnan",      "Isfinite",   "Lgamma",
          "Log",      "Log1p",    "LogicalNot", "Neg",        "Reciprocal",
          "Relu",     "Relu6",    "Rint",       "Round",      "Selu",
          "Rsqrt",    "Sigmoid",  "Sign",       "Sin",        "SinH",
          "Softplus", "Softsign", "Sqrt",       "Square",     "Tan",
          "Tanh",
      }));
  return kElementWiseOps->count(node.op()) > 0 ||
         IsValueAndOrderAndShapePreserving(node);
}

bool HasOpDef(const NodeDef& node) {
   std::vector<std::string> mht_200_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_200(mht_200_v, 1643, "", "./tensorflow/core/grappler/op_types.cc", "HasOpDef");

  const OpDef* op_def = nullptr;
  return OpRegistry::Global()->LookUpOpDef(node.op(), &op_def).ok();
}

bool IsIdempotent(const NodeDef& node) {
   std::vector<std::string> mht_201_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_201(mht_201_v, 1651, "", "./tensorflow/core/grappler/op_types.cc", "IsIdempotent");

  return IsValueAndOrderAndShapePreserving(node) && IsFreeOfSideEffect(node) &&
         !ModifiesFrameInfo(node);
}

bool NeverForwardsInputs(const NodeDef& node) {
   std::vector<std::string> mht_202_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_202(mht_202_v, 1659, "", "./tensorflow/core/grappler/op_types.cc", "NeverForwardsInputs");

  static const gtl::FlatSet<string>* const kNonForwardingOps = CHECK_NOTNULL(
      (new gtl::FlatSet<string>{"ArgMax",
                                "ArgMin",
                                "AudioSpectrogram",
                                "AvgPool",
                                "BatchMatMul",
                                "BatchMatMulV2",
                                "BatchNormWithGlobalNormalization",
                                "BatchToSpace",
                                "BatchToSpaceND",
                                "Bincount",
                                "BroadcastArgs",
                                "BroadcastGradientArgs",
                                "Bucketize",
                                "CTCBeamSearchDecoder",
                                "CTCGreedyDecoder",
                                "CTCLoss",
                                "CompareAndBitpack",
                                "ComplexAbs",
                                "Concat",
                                "ConcatOffset",
                                "ConcatV2",
                                "Conv2D",
                                "Copy",
                                "CopyHost",
                                "Cross",
                                "CudnnRNN",
                                "CudnnRNNBackprop",
                                "CudnnRNNBackpropV2",
                                "CudnnRNNBackpropV3",
                                "CudnnRNNCanonicalToParams",
                                "CudnnRNNCanonicalToParamsV2",
                                "CudnnRNNParamsSize",
                                "CudnnRNNParamsToCanonical",
                                "CudnnRNNParamsToCanonicalV2",
                                "CudnnRNNV2",
                                "CudnnRNNV3",
                                "CumProd",
                                "CumSum",
                                "DebugNanCount",
                                "DebugNumericSummary",
                                "DecodeProtoV2",
                                "DecodeWav",
                                "DeepCopy",
                                "DepthToSpace",
                                "Dequantize",
                                "Diag",
                                "DiagPart",
                                "EditDistance",
                                "Empty",
                                "EncodeProtoV2",
                                "EncodeWav",
                                "ExtractImagePatches",
                                "ExtractVolumePatches",
                                "Fill",
                                "Gather",
                                "GatherNd",
                                "GatherV2",
                                "HistogramFixedWidth",
                                "InvertPermutation",
                                "IsInf",
                                "IsNan",
                                "Isfinite",
                                "LinSpace",
                                "LowerBound",
                                "MatMul",
                                "MatrixDiag",
                                "MatrixDiagPart",
                                "MatrixDiagPartV2",
                                "MatrixDiagV2",
                                "Mfcc",
                                "Multinomial",
                                "OneHot",
                                "Pack",
                                "ParameterizedTruncatedNormal",
                                "PopulationCount",
                                "RandomGamma",
                                "RandomPoisson",
                                "RandomPoissonV2",
                                "RandomStandardNormal",
                                "RandomUniform",
                                "RandomUniformInt",
                                "Range",
                                "Rank",
                                "RequantizationRange",
                                "Requantize",
                                "ReverseSequence",
                                "Shape",
                                "ShapeN",
                                "Size",
                                "SpaceToBatch",
                                "SpaceToBatchND",
                                "SpaceToDepth",
                                "SparseMatMul",
                                "Split",
                                "SplitV",
                                "TruncatedNormal",
                                "Unique",
                                "UniqueV2",
                                "UniqueWithCounts",
                                "UniqueWithCountsV2",
                                "Unpack",
                                "UnravelIndex",
                                "UpperBound",
                                "Where"}));
  const string& op_name = node.op();
  return kNonForwardingOps->count(op_name) > 0 ||
         absl::StrContains(op_name, "Segment") ||
         absl::StartsWith(op_name, "Quantize");
}

bool IsXlaLaunch(const NodeDef& node) {
   std::vector<std::string> mht_203_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSop_typesDTcc mht_203(mht_203_v, 1774, "", "./tensorflow/core/grappler/op_types.cc", "IsXlaLaunch");
 return node.op() == "XlaLaunch"; }

}  // namespace grappler
}  // end namespace tensorflow
