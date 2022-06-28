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
class MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc() {
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
#include <ctype.h>

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "flatbuffers/minireflect.h"  // from @flatbuffers
#include "tensorflow/lite/schema/reflection/schema_generated.h"

namespace tflite {
namespace {
// This is generated by grepping
//  cat  third_party/tensorflow/lite/c/builtin_op_data.h | grep "^} TfLite" |
//  sed 's/^} \(TfLite.*\)Params;/\1Params/g' | grep -v "^}" | sed
//  's/\(.*\)/"\1",/g' | sort
static const char* param_structs[] = {"TfLiteAddParams",
                                      "TfLiteArgMaxParams",
                                      "TfLiteArgMinParams",
                                      "TfLiteBatchMatMulParams",
                                      "TfLiteBatchToSpaceNDParams",
                                      "TfLiteBidirectionalSequenceLSTMParams",
                                      "TfLiteBidirectionalSequenceRNNParams",
                                      "TfLiteBucketizeParams",
                                      "TfLiteCastParams",
                                      "TfLiteConcatenationParams",
                                      "TfLiteConvParams",
                                      "TfLiteDepthwiseConvParams",
                                      "TfLiteDivParams",
                                      "TfLiteDynamicUpdateSliceParams",
                                      "TfLiteEmbeddingLookupSparseParams",
                                      "TfLiteFakeQuantParams",
                                      "TfLiteFullyConnectedParams",
                                      "TfLiteGatherParams",
                                      "TfLiteGeluParams",
                                      "TfLiteIfParams",
                                      "TfLiteL2NormParams",
                                      "TfLiteLeakyReluParams",
                                      "TfLiteLocalResponseNormParams",
                                      "TfLiteLSHProjectionParams",
                                      "TfLiteLSTMParams",
                                      "TfLiteMirrorPaddingParams",
                                      "TfLiteMulParams",
                                      "TfLiteOneHotParams",
                                      "TfLitePackParams",
                                      "TfLitePadParams",
                                      "TfLitePadV2Params",
                                      "TfLitePoolParams",
                                      "TfLiteRandomParams",
                                      "TfLiteReducerParams",
                                      "TfLiteReshapeParams",
                                      "TfLiteResizeBilinearParams",
                                      "TfLiteResizeNearestNeighborParams",
                                      "TfLiteRNNParams",
                                      "TfLiteSequenceRNNParams",
                                      "TfLiteShapeParams",
                                      "TfLiteSkipGramParams",
                                      "TfLiteSoftmaxParams",
                                      "TfLiteSpaceToBatchNDParams",
                                      "TfLiteSpaceToDepthParams",
                                      "TfLiteDepthToSpaceParams",
                                      "TfLiteSparseToDenseParams",
                                      "TfLiteSplitParams",
                                      "TfLiteSplitVParams",
                                      "TfLiteSqueezeParams",
                                      "TfLiteStridedSliceParams",
                                      "TfLiteSubParams",
                                      "TfLiteSVDFParams",
                                      "TfLiteTransposeConvParams",
                                      "TfLiteTransposeParams",
                                      "TfLiteUnidirectionalSequenceLSTMParams",
                                      "TfLiteUniqueParams",
                                      "TfLiteUnpackParams",
                                      "TfLiteReverseSequenceParams",
                                      "TfLiteWhileParams",
                                      "TfLiteCumsumParams",
                                      "TfLiteCallOnceParams",
                                      "TfLiteConv3DParams",
                                      "TfLiteHashtableParams",
                                      "TfLiteHashtableFindParams",
                                      "TfLiteHashtableImportParams",
                                      "TfLiteHashtableSizeParams",
                                      "TfLiteConv3DTransposeParams",
                                      "TfLiteVarHandleParams",
                                      nullptr};
}  // namespace

// Get rid of all underscores and make everything lower case to make name
// matching work for stuff like 3D vs 3d or RNN vs Rnn.
std::string ToCollapsed(const std::string& in) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("in: \"" + in + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_0(mht_0_v, 273, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "ToCollapsed");

  const char* s = in.c_str();
  bool first = true;
  std::string out;
  while (*s != '\0') {
    if (*s == '_') {
      first = true;
    } else if (first) {
      out.push_back(tolower(*s));
      first = false;
    } else {
      out.push_back(tolower(*s));
    }
    s++;
  }
  return out;
}

// A collection of information about builtin ops.
class OpOptionData {
 public:
  OpOptionData() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_1(mht_1_v, 297, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "OpOptionData");

    BuildOpList();
    BuildOptionToTypeFunctionMap();
    BuildOpToOptionMap();
  }

  // A list of builtin operations
  const std::vector<std::string>& ops() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_2(mht_2_v, 307, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "ops");
 return ops_; }
  // Maps from operation name to option name (i.e. 'ADD' to 'AddOptions')
  const std::unordered_map<std::string, std::string>& op_to_option() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_3(mht_3_v, 312, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "op_to_option");

    return op_to_option_;
  }
  // Maps from option to C struct i.e. 'AddOptions' -> 'TfLiteAddOptions'
  const std::unordered_map<std::string, std::string>& option_to_struct() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_4(mht_4_v, 319, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "option_to_struct");

    return option_to_struct_;
  }
  // Maps from option to a flatbuffer type function that describes that option.
  const std::unordered_map<std::string, flatbuffers::TypeFunction>&
  option_to_type_function() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_5(mht_5_v, 327, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "option_to_type_function");

    return option_to_type_function_;
  }

 private:
  void BuildOpList() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_6(mht_6_v, 335, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "BuildOpList");

    for (const char* const* curr = EnumNamesBuiltinOperator(); *curr != nullptr;
         ++curr) {
      if (strlen(*curr) != 0) ops_.push_back(*curr);
    }
  }

  void BuildOptionToTypeFunctionMap() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_7(mht_7_v, 345, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "BuildOptionToTypeFunctionMap");

    auto d = tflite::BuiltinOptionsTypeTable();
    for (int i = 0; i < d->num_elems; i++) {
      flatbuffers::TypeCode code = d->type_codes[i];
      if (code.sequence_ref != -1) {
        option_to_type_function_.insert(
            std::make_pair(d->names[i], d->type_refs[code.sequence_ref]));
      }
    }
  }

  void BuildOpToOptionMap() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_8(mht_8_v, 359, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "BuildOpToOptionMap");

    // Manually specified mappings between ops and options
    op_to_option_["REDUCE_MAX"] = "ReducerOptions";
    op_to_option_["REDUCE_MIN"] = "ReducerOptions";
    op_to_option_["REDUCE_ANY"] = "ReducerOptions";
    op_to_option_["REDUCE_ALL"] = "ReducerOptions";
    op_to_option_["SUM"] = "ReducerOptions";
    op_to_option_["REDUCE_MAX"] = "ReducerOptions";
    op_to_option_["REDUCE_PROD"] = "ReducerOptions";
    op_to_option_["MEAN"] = "ReducerOptions";
    op_to_option_["L2_POOL_2D"] = "Pool2DOptions";
    op_to_option_["AVERAGE_POOL_2D"] = "Pool2DOptions";
    op_to_option_["MAX_POOL_2D"] = "Pool2DOptions";
    op_to_option_["L2_NORMALIZATION"] = "L2NormOptions";
    op_to_option_["UNIDIRECTIONAL_SEQUENCE_RNN"] = "SequenceRNNOptions";
    op_to_option_["MAXIMUM"] = "MaximumMinimumOptions";
    op_to_option_["MINIMUM"] = "MaximumMinimumOptions";
    op_to_option_["CONV_3D_TRANSPOSE"] = "Conv3DOptions";
    op_to_option_["RANDOM_STANDARD_NORMAL"] = "RandomOptions";
    op_to_option_["RANDOM_UNIFORM"] = "RandomOptions";
    op_to_option_["MULTINOMIAL"] = "RandomOptions";

    // These operators are not real ones.
    op_to_option_["CUSTOM"] = "";    // TODO(aselle): maybe something else.
    op_to_option_["DELEGATE"] = "";  // TODO(aselle): maybe something else.
    op_to_option_["PLACEHOLDER_FOR_GREATER_OP_CODES"] = "";

    // Manually specified mappings between ops to "none" options -- these are
    // ops without a corresponding Options message in schema as yet. If these
    // options do get assigned an Options message in future, they need to be
    // updated here as well.
    op_to_option_["EMBEDDING_LOOKUP"] = "";
    op_to_option_["FLOOR"] = "";
    op_to_option_["CEIL"] = "";
    op_to_option_["HASHTABLE_LOOKUP"] = "";
    op_to_option_["LOGISTIC"] = "";
    op_to_option_["RELU"] = "";
    op_to_option_["RELU_N1_TO_1"] = "";
    op_to_option_["RELU6"] = "";
    op_to_option_["ROUND"] = "";
    op_to_option_["TANH"] = "";
    op_to_option_["PRELU"] = "";
    op_to_option_["SIN"] = "";
    op_to_option_["LOG"] = "";
    op_to_option_["SQRT"] = "";
    op_to_option_["RSQRT"] = "";
    op_to_option_["ELU"] = "";
    op_to_option_["REVERSE_SEQUENCE"] = "";
    op_to_option_["REAL"] = "";
    op_to_option_["IMAG"] = "";
    op_to_option_["COMPLEX_ABS"] = "";
    op_to_option_["BROADCAST_ARGS"] = "";
    op_to_option_["GELU"] = "";
    op_to_option_["DYNAMIC_UPDATE_SLICE"] = "";

    // TODO(aselle): These are undesirable hacks. Consider changing C structs
    option_to_struct_["Pool2DOptions"] = "TfLitePoolParams";
    option_to_struct_["Conv2DOptions"] = "TfLiteConvParams";
    option_to_struct_["DepthwiseConv2DOptions"] = "TfLiteDepthwiseConvParams";
    option_to_struct_["LocalResponseNormalizationOptions"] =
        "TfLiteLocalResponseNormParams";
    option_to_struct_["MirrorPadOptions"] = "TfLiteMirrorPaddingParams";
    // Now for every op, try to find an option.
    bool fatal = false;
    for (const auto& op_name : ops_) {
      auto d = tflite::BuiltinOptionsTypeTable();
      std::string collapsed_option_name_guess =
          ToCollapsed(op_name) + "options";
      // O(n^2) but not that big of n.
      for (int i = 0; i < d->num_elems; i++) {
        std::string option_name = d->names[i];
        std::string collapsed_option_name = ToCollapsed(option_name);
        if (collapsed_option_name_guess == collapsed_option_name) {
          op_to_option_.insert(std::make_pair(op_name, option_name));
          break;
        }
      }
      auto it = op_to_option_.find(op_name);
      if (it == op_to_option_.end()) {
        std::cerr << "Didn't find option for  " << op_name << std::endl;
        fatal = true;
      } else if (!it->second.empty()) {
        std::string option_name = it->second;

        if (option_to_struct_.find(option_name) == option_to_struct_.end()) {
          bool param_struct_found = false;
          std::string params_guess = std::string("TfLite") + option_name;
          size_t start = params_guess.find("Options");
          size_t len = strlen("Options");
          params_guess.replace(start, len, "Params");
          for (auto* param = param_structs; *param != nullptr; param++) {
            if (*param == params_guess) {
              param_struct_found = true;
              break;
            }
          }
          if (!param_struct_found) {
            std::cerr << "Failed to get param struct for option " << option_name
                      << std::endl;
          } else {
            option_to_struct_.insert(std::make_pair(option_name, params_guess));
          }
        }
      }
    }
    if (fatal) {
      exit(1);
    }
  }

 private:
  std::vector<std::string> ops_;
  std::unordered_map<std::string, std::string> op_to_option_;
  std::unordered_map<std::string, std::string> option_to_struct_;
  std::unordered_map<std::string, flatbuffers::TypeFunction>
      option_to_type_function_;
};

void GenerateImportForResizeBilinearOp(FILE* fp) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_9(mht_9_v, 480, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "GenerateImportForResizeBilinearOp");

  fprintf(fp,
          "  case BuiltinOperator_RESIZE_BILINEAR:  {\n"
          "    const auto* params = reinterpret_cast<const "
          "TfLiteResizeBilinearParams*>(builtin_op_data);\n"
          "    auto union_type = CreateResizeBilinearOptions(*fbb, "
          "params->align_corners, params->half_pixel_centers).Union();\n"
          "    return std::make_pair(BuiltinOptions_ResizeBilinearOptions, "
          "union_type);\n"
          "  }\n  break;\n");
}

void GenerateImportForVarHandleOp(FILE* fp) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_10(mht_10_v, 495, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "GenerateImportForVarHandleOp");

  fprintf(fp,
          "  case BuiltinOperator_VAR_HANDLE:  {\n"
          "    const auto* params = reinterpret_cast<const "
          "TfLiteVarHandleParams*>(builtin_op_data);\n"
          "    auto union_type = CreateVarHandleOptions(*fbb, "
          "fbb->CreateString(params->container), "
          "fbb->CreateString(params->shared_name)).Union();\n"
          "    return std::make_pair(BuiltinOptions_VarHandleOptions, "
          "union_type);\n"
          "  }\n  break;\n");
}

// Reshape Op infers output shape either from Parameter or from shape tensor
// that's is an additional input. When we have this additional shape tensor as
// input we don't have the parameter present in this layer. In case of more than
// one input and the shape parameter does not have a valid value, we import an
// empty vector for the parameters.
void GenerateImportForReshapeOp(FILE* fp) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_11(mht_11_v, 516, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "GenerateImportForReshapeOp");

  fprintf(fp,
          "  case BuiltinOperator_RESHAPE:  {\n"
          "    const auto* params = reinterpret_cast<const "
          "TfLiteReshapeParams*>(builtin_op_data);\n"
          "    flatbuffers::Offset<void> union_type;\n"
          "    if (node.inputs->size > 1 && (params->num_dimensions <= 0 || "
          "params->num_dimensions > TFLITE_RESHAPE_PARAMS_MAX_DIMENSION_COUNT))"
          " {\n"
          "      union_type = CreateReshapeOptions(*fbb).Union();\n"
          "    } else {\n"
          "      auto val0 = fbb->CreateVector(std::vector<int>(params->shape, "
          "params->shape + params->num_dimensions));\n"
          "      union_type = CreateReshapeOptions(*fbb, "
          "val0).Union();\n"
          "    }\n"
          "    return std::make_pair(BuiltinOptions_ReshapeOptions, "
          "union_type);\n"
          "  }\n  break;\n");
}

void GenerateImportForOp(FILE* fp, const std::string& op_name,
                         const std::string& option_name,
                         const std::string& option_type,
                         const flatbuffers::TypeTable* options,
                         const std::string& struct_name) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("op_name: \"" + op_name + "\"");
   mht_12_v.push_back("option_name: \"" + option_name + "\"");
   mht_12_v.push_back("option_type: \"" + option_type + "\"");
   mht_12_v.push_back("struct_name: \"" + struct_name + "\"");
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_12(mht_12_v, 548, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "GenerateImportForOp");

  // Special-case ResizeBilinear which has some deprecated fields.
  if (struct_name == "TfLiteResizeBilinearParams") {
    GenerateImportForResizeBilinearOp(fp);
    return;
  }

  if (struct_name == "TfLiteVarHandleParams") {
    GenerateImportForVarHandleOp(fp);
    return;
  }

  // Special case Reshape that may have 'new_shape' field missing from the
  // parameters.
  if (struct_name == "TfLiteReshapeParams") {
    GenerateImportForReshapeOp(fp);
    return;
  }

  fprintf(fp, "  case BuiltinOperator_%s:  {\n", op_name.c_str());
  if (options->num_elems != 0) {
    fprintf(fp,
            "    const auto* params = reinterpret_cast<const "
            "%s*>(builtin_op_data);\n",
            struct_name.c_str());
  }

  for (size_t i = 0; i < options->num_elems; i++) {
    std::string elem_name = options->names[i];
    bool is_int_vector = false;
    bool is_float_vector = false;
    std::string vector_name = elem_name;
    std::string vector_size;
    // TODO(aselle): Irregular naming in builtins
    if (elem_name == "fused_activation_function")
      elem_name = "activation";
    else if (elem_name == "stride_w")
      elem_name = "stride_width";
    else if (elem_name == "stride_h")
      elem_name = "stride_height";
    else if (elem_name == "stride_d")
      elem_name = "stride_depth";
    else if (elem_name == "dilation_h_factor")
      elem_name = "dilation_height_factor";
    else if (elem_name == "dilation_w_factor")
      elem_name = "dilation_width_factor";
    else if (elem_name == "dilation_d_factor")
      elem_name = "dilation_depth_factor";
    else if (elem_name == "idx_out_type")
      elem_name = "index_out_type";

    // Vector fields treated specially.
    if (elem_name == "new_shape") {
      is_int_vector = true;
      vector_name = "shape";
      vector_size = "num_dimensions";
    } else if (elem_name == "squeeze_dims") {
      is_int_vector = true;
      vector_size = "num_squeeze_dims";
    } else if (elem_name == "boundaries") {
      is_float_vector = true;
      vector_size = "num_boundaries";
    }

    if (is_int_vector) {
      fprintf(fp,
              "    auto val%zu = fbb->CreateVector("
              "std::vector<int>(params->%s, params->%s + params->%s));\n",
              i, vector_name.c_str(), vector_name.c_str(), vector_size.c_str());
      continue;
    }

    if (is_float_vector) {
      fprintf(fp,
              "    auto val%zu = fbb->CreateVector("
              "std::vector<float>(params->%s, params->%s + params->%s));\n",
              i, vector_name.c_str(), vector_name.c_str(), vector_size.c_str());
      continue;
    }

    flatbuffers::TypeCode code = options->type_codes[i];
    auto contained_type = code.sequence_ref != -1
                              ? options->type_refs[code.sequence_ref]
                              : nullptr;
    std::string mapper = "";
    if (contained_type == TensorTypeTypeTable) {
      mapper = "TfLiteTypeToSchemaType";
    } else if (contained_type == ActivationFunctionTypeTypeTable) {
      mapper = "TfLiteActivationToSchemaActivation";
    } else if (contained_type == PaddingTypeTable) {
      mapper = "TfLitePaddingToSchemaPadding";
    } else if (contained_type == FullyConnectedOptionsWeightsFormatTypeTable) {
      mapper = "FullyConnectedOptionsWeightsFormatToSchema";
    } else if (contained_type == LSTMKernelTypeTypeTable) {
      mapper = "LSTMKernelTypeToSchema";
    } else if (contained_type == LSHProjectionTypeTypeTable) {
      mapper = "LSHProjectionTypeToSchema";
    } else if (contained_type == MirrorPadModeTypeTable) {
      mapper = "MirrorPaddingModeToSchema";
    } else if (contained_type == CombinerTypeTypeTable) {
      mapper = "CombinerTypeToSchema";
    }

    fprintf(fp,
            "    auto val%zu = "
            "%s(params->%s);\n",
            i, mapper.c_str(), elem_name.c_str());
  }
  fprintf(fp, "    auto union_type = Create%s(*fbb", option_name.c_str());
  for (size_t i = 0; i < options->num_elems; i++) {
    fprintf(fp, ", val%zu", i);
  }
  fprintf(fp, ").Union();\n");
  fprintf(fp, "    return std::make_pair(%s, union_type);\n",
          option_type.c_str());
  fprintf(fp, "  }\n  break;\n");
}

void GenerateImport(OpOptionData* option, FILE* fp) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_13(mht_13_v, 669, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "GenerateImport");

  std::unordered_set<std::string> ignores;
  ignores.insert("CONCAT_EMBEDDINGS");
  ignores.insert("CALL");

  // Allow any op that doesn't have an options struct to be blocked
  // together
  for (const auto& op_name : option->ops()) {
    auto option_it = option->op_to_option().find(op_name);
    if (!option_it->second.empty() && ignores.find(op_name) == ignores.end())
      continue;
    fprintf(fp, "  case BuiltinOperator_%s:\n", op_name.c_str());
  }
  fprintf(fp,
          "    return std::make_pair(BuiltinOptions_NONE, "
          "flatbuffers::Offset<void>());\n    break;\n");

  // Iterate over each ops
  for (const auto& op_name : option->ops()) {
    if (ignores.find(op_name) != ignores.end()) continue;
    // Get to the option and struct names, continuing if not found.
    auto option_it = option->op_to_option().find(op_name);
    if (option_it->second.empty()) continue;
    std::string option_name = option_it->second;
    std::string option_type = "BuiltinOptions_" + option_name;
    auto option_func_it = option->option_to_type_function().find(option_name);
    if (option_func_it == option->option_to_type_function().end()) continue;
    auto struct_name_it = option->option_to_struct().find(option_name);
    if (struct_name_it == option->option_to_struct().end()) {
      // If no C struct, then it better have no arguments.
      auto type_info = option_func_it->second();
      if (type_info->num_elems != 0) {
        // We have non-zero arguments in the schema, this means there
        // should be a struct.
        fprintf(stderr,
                "Op %s uses option struct %s which has no builtin struct\n",
                op_name.c_str(), option_name.c_str());
        exit(1);
      }
      fprintf(fp, "  case BuiltinOperator_%s:\n", op_name.c_str());
      fprintf(fp, "    return std::make_pair(%s, Create%s(*fbb).Union());",
              option_type.c_str(), option_name.c_str());
    } else {
      // If C struct, then we need to assign all properties
      auto struct_name = struct_name_it->second;
      GenerateImportForOp(fp, op_name, option_name, option_type,
                          option_func_it->second(), struct_name);
    }
  }
  // TODO(aselle): Handle unhandled cases more gracefully.
  fprintf(fp,
          "default:    return std::make_pair(BuiltinOptions_NONE, "
          "flatbuffers::Offset<void>());\n    break;\n");
}

}  // namespace tflite

int main(int argc, char* argv[]) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSserializationPSoption_writer_generatorDTcc mht_14(mht_14_v, 729, "", "./tensorflow/lite/tools/serialization/option_writer_generator.cc", "main");

  tflite::OpOptionData option;
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <fname out>\n", argv[0]);
    return 1;
  }
  FILE* fp = fopen(argv[1], "w");
  tflite::GenerateImport(&option, fp);
  fclose(fp);
}
