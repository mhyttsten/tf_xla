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
class MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_gradient_exclusionsDTcc {
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
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_gradient_exclusionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_gradient_exclusionsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Inputs/Outputs exclusion lists for GradientTape.
//
// This file is MACHINE GENERATED! Do not edit.
// Generated by: tensorflow/python/eager/gen_gradient_input_output_exclusions.py

#include "tensorflow/python/eager/pywrap_gradient_exclusions.h"

#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"

using tensorflow::string;

namespace {
// Keep static data in a format that's easy to init statically.
struct OpIndexInfo {
  const char *op_name;
  int num_indices;
  std::array<int, 4> unused_indices;
};

// Helper function to initialize FlatMap<string,FlatSet> from OpIndexInfo.
template <typename T>
auto OpGradientInfoInit(const T &a) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSpythonPSeagerPSpywrap_gradient_exclusionsDTcc mht_0(mht_0_v, 208, "", "./tensorflow/python/eager/pywrap_gradient_exclusions.cc", "OpGradientInfoInit");

  auto *m = new tensorflow::gtl::FlatMap<string, tensorflow::gtl::FlatSet<int>>;
  for (const auto &item : a) {
    m->emplace(string(item.op_name),
               tensorflow::gtl::FlatSet<int>(
                   item.unused_indices.begin(),
                   item.unused_indices.begin() + item.num_indices));
  }
  return m;
}
}  // namespace

absl::optional<tensorflow::gtl::FlatSet<int>> OpGradientUnusedInputIndices(
    const tensorflow::string &op_name) {
  static std::array<OpIndexInfo, 363> a = {{
      {"Acosh"},
      {"AllToAll", 1, {0}},
      {"ApproximateEqual"},
      {"ArgMax"},
      {"ArgMin"},
      {"AsString"},
      {"Asinh"},
      {"Assign"},
      {"AssignAdd"},
      {"AssignSub"},
      {"AudioSummary"},
      {"AudioSummaryV2"},
      {"AvgPool3DGrad", 1, {1}},
      {"AvgPoolGrad", 1, {1}},
      {"BatchNormWithGlobalNormalization", 1, {3}},
      {"BatchToSpace", 1, {0}},
      {"BatchToSpaceND", 1, {0}},
      {"BiasAdd"},
      {"BiasAddV1"},
      {"BitwiseAnd"},
      {"BitwiseOr"},
      {"BitwiseXor"},
      {"BroadcastGradientArgs"},
      {"CSRSparseMatrixToSparseTensor"},
      {"CTCBeamSearchDecoder"},
      {"CTCGreedyDecoder"},
      {"CTCLoss"},
      {"CTCLossV2"},
      {"Ceil"},
      {"CheckNumerics"},
      {"CheckNumericsV2"},
      {"Cholesky"},
      {"CollectivePermute", 1, {0}},
      {"CompositeTensorVariantFromComponents"},
      {"CompositeTensorVariantToComponents"},
      {"Conj"},
      {"ConjugateTranspose", 1, {0}},
      {"Const"},
      {"Conv2DBackpropFilter", 1, {1}},
      {"Conv2DBackpropInput", 1, {0}},
      {"Conv3DBackpropFilterV2", 1, {1}},
      {"Conv3DBackpropInputV2", 1, {0}},
      {"CropAndResize", 1, {3}},
      {"CrossReplicaSum", 1, {0}},
      {"Cumsum", 1, {0}},
      {"DebugGradientIdentity"},
      {"DebugGradientRefIdentity"},
      {"DebugIdentityV2"},
      {"DecodeBase64"},
      {"DecodePaddedRaw"},
      {"DecodeProtoV2"},
      {"DecodeRaw"},
      {"DeleteSessionTensor"},
      {"DenseToCSRSparseMatrix"},
      {"DenseToDenseSetOperation"},
      {"DenseToSparseSetOperation"},
      {"DepthToSpace"},
      {"DepthwiseConv2dNativeBackpropFilter", 1, {1}},
      {"DepthwiseConv2dNativeBackpropInput", 1, {0}},
      {"Diag"},
      {"DiagPart"},
      {"DrawBoundingBoxes"},
      {"EditDistance"},
      {"Elu"},
      {"EncodeBase64"},
      {"EnsureShape"},
      {"Enter"},
      {"Equal"},
      {"Erfinv"},
      {"Exit"},
      {"Exp"},
      {"ExpandDims", 1, {1}},
      {"ExtractGlimpse"},
      {"FFT"},
      {"FFT2D"},
      {"FFT3D"},
      {"Fill"},
      {"FixedLengthRecordReader"},
      {"Floor"},
      {"FloorDiv"},
      {"FusedBatchNorm", 1, {2}},
      {"FusedBatchNormGradV3", 1, {5}},
      {"FusedBatchNormV2", 1, {2}},
      {"FusedBatchNormV3", 1, {2}},
      {"GenerateBoundingBoxProposals"},
      {"GenerateVocabRemapping"},
      {"GetSessionHandle"},
      {"GetSessionHandleV2"},
      {"GetSessionTensor"},
      {"Greater"},
      {"GreaterEqual"},
      {"HSVToRGB"},
      {"HashTable"},
      {"HashTableV2"},
      {"HistogramSummary"},
      {"IFFT"},
      {"IFFT2D"},
      {"IFFT3D"},
      {"Identity"},
      {"IdentityN"},
      {"IdentityReader"},
      {"Imag"},
      {"ImageProjectiveTransformV2", 1, {2}},
      {"ImageProjectiveTransformV3", 2, {2, 3}},
      {"ImageSummary"},
      {"InitializeTable"},
      {"InitializeTableFromTextFile"},
      {"InitializeTableFromTextFileV2"},
      {"InitializeTableV2"},
      {"Inv"},
      {"Invert"},
      {"InvertPermutation"},
      {"IsotonicRegression"},
      {"LMDBReader"},
      {"LeakyReluGrad", 1, {0}},
      {"LeftShift"},
      {"Less"},
      {"LessEqual"},
      {"LinSpace"},
      {"LoadAndRemapMatrix"},
      {"LogSoftmax"},
      {"LogicalAnd"},
      {"LogicalNot"},
      {"LogicalOr"},
      {"LookupTableFind"},
      {"LookupTableFindV2"},
      {"LookupTableInsert"},
      {"LookupTableInsertV2"},
      {"LookupTableSize"},
      {"LookupTableSizeV2"},
      {"LoopCond"},
      {"MatrixBandPart", 1, {0}},
      {"MatrixDiag"},
      {"MatrixDiagPartV2", 1, {2}},
      {"MatrixDiagPartV3", 1, {2}},
      {"MatrixDiagV2", 4, {0, 2, 3, 4}},
      {"MatrixDiagV3", 4, {0, 2, 3, 4}},
      {"MatrixInverse"},
      {"MatrixSetDiagV2", 1, {0}},
      {"MatrixSetDiagV3", 1, {0}},
      {"MatrixSolve", 1, {1}},
      {"MatrixSquareRoot"},
      {"MaxPool3DGrad", 1, {2}},
      {"MaxPool3DGradGrad", 1, {2}},
      {"MaxPoolGrad", 1, {2}},
      {"MaxPoolGradGrad", 1, {2}},
      {"MaxPoolGradV2", 1, {2}},
      {"MirrorPad", 1, {0}},
      {"MirrorPadGrad", 1, {0}},
      {"Multinomial"},
      {"MutableDenseHashTable"},
      {"MutableDenseHashTableV2"},
      {"MutableHashTable"},
      {"MutableHashTableOfTensors"},
      {"MutableHashTableOfTensorsV2"},
      {"MutableHashTableV2"},
      {"NcclAllReduce"},
      {"NcclBroadcast"},
      {"Ndtri"},
      {"Neg"},
      {"NextIteration"},
      {"NonMaxSuppression"},
      {"NonMaxSuppressionV2"},
      {"NonMaxSuppressionWithOverlaps"},
      {"NotEqual"},
      {"NthElement", 1, {1}},
      {"OneHot"},
      {"OnesLike"},
      {"OptionalGetValue"},
      {"Pack"},
      {"ParameterizedTruncatedNormal"},
      {"ParseTensor"},
      {"PlaceholderWithDefault"},
      {"PopulationCount"},
      {"PreventGradient"},
      {"QuantizeAndDequantize"},
      {"QuantizeAndDequantizeV2"},
      {"QuantizeAndDequantizeV3"},
      {"QuantizeAndDequantizeV4Grad", 1, {3}},
      {"QueueClose"},
      {"QueueDequeue"},
      {"QueueDequeueMany"},
      {"QueueDequeueUpTo"},
      {"QueueSize"},
      {"RaggedRange"},
      {"RandomCrop"},
      {"RandomIndexShuffle"},
      {"RandomStandardNormal"},
      {"RandomUniform"},
      {"Range"},
      {"Rank"},
      {"ReadVariableOp"},
      {"ReaderNumRecordsProduced"},
      {"ReaderNumWorkUnitsCompleted"},
      {"ReaderRead"},
      {"ReaderReadUpTo"},
      {"ReaderReset"},
      {"ReaderRestoreState"},
      {"ReaderSerializeState"},
      {"Real"},
      {"Reciprocal"},
      {"ReduceJoin"},
      {"RefEnter"},
      {"RefExit"},
      {"RefIdentity"},
      {"RefNextIteration"},
      {"RegexReplace"},
      {"Relu"},
      {"Relu6"},
      {"Relu6Grad", 1, {0}},
      {"ReluGrad", 1, {0}},
      {"Reshape", 1, {1}},
      {"ResizeBicubic", 1, {1}},
      {"ResizeBilinear", 1, {1}},
      {"ResizeNearestNeighbor", 1, {1}},
      {"Reverse", 1, {0}},
      {"ReverseSequence", 1, {0}},
      {"ReverseV2", 1, {0}},
      {"RightShift"},
      {"Rint"},
      {"Roll", 1, {0}},
      {"Round"},
      {"Rsqrt"},
      {"SampleDistortedBoundingBox"},
      {"SampleDistortedBoundingBoxV2"},
      {"ScalarSummary"},
      {"ScaleAndTranslate", 1, {1}},
      {"ScatterAdd"},
      {"ScatterDiv"},
      {"ScatterMul"},
      {"ScatterNd", 2, {1, 2}},
      {"ScatterNdAdd"},
      {"ScatterNdNonAliasingAdd", 2, {0, 2}},
      {"ScatterNdSub"},
      {"ScatterNdUpdate"},
      {"ScatterSub"},
      {"SdcaFprint"},
      {"SegmentSum", 1, {0}},
      {"Select", 1, {2}},
      {"Selu"},
      {"SerializeTensor"},
      {"SetSize"},
      {"Shape"},
      {"Sigmoid"},
      {"Size"},
      {"Slice", 1, {2}},
      {"Softmax"},
      {"SoftmaxCrossEntropyWithLogits", 1, {1}},
      {"SpaceToBatch", 1, {0}},
      {"SpaceToBatchND", 1, {0}},
      {"SpaceToDepth"},
      {"SparseAdd", 3, {2, 5, 6}},
      {"SparseAddGrad"},
      {"SparseDenseCwiseAdd"},
      {"SparseFillEmptyRows"},
      {"SparseMatrixMul"},
      {"SparseMatrixNNZ"},
      {"SparseMatrixSoftmax"},
      {"SparseMatrixTranspose"},
      {"SparseMatrixZeros"},
      {"SparseReduceSum", 1, {1}},
      {"SparseReorder", 1, {1}},
      {"SparseSegmentMeanWithNumSegments", 1, {3}},
      {"SparseSegmentSqrtNWithNumSegments", 1, {3}},
      {"SparseSegmentSumWithNumSegments", 1, {3}},
      {"SparseSlice", 2, {2, 4}},
      {"SparseSoftmax", 1, {1}},
      {"SparseSoftmaxCrossEntropyWithLogits", 1, {1}},
      {"SparseSparseMaximum"},
      {"SparseSparseMinimum"},
      {"SparseTensorDenseAdd", 3, {1, 2, 3}},
      {"SparseTensorToCSRSparseMatrix"},
      {"SparseToSparseSetOperation"},
      {"Split", 1, {1}},
      {"Sqrt"},
      {"SqrtGrad", 1, {1}},
      {"Stack"},
      {"StackClose"},
      {"StackPop"},
      {"StackPush"},
      {"StatelessCase"},
      {"StatelessMultinomial"},
      {"StatelessParameterizedTruncatedNormal", 1, {1}},
      {"StatelessRandomBinomial"},
      {"StatelessRandomGammaV2", 1, {1}},
      {"StatelessRandomNormal"},
      {"StatelessRandomNormalV2"},
      {"StatelessRandomPoisson"},
      {"StatelessRandomUniform"},
      {"StatelessRandomUniformFullInt"},
      {"StatelessRandomUniformFullIntV2"},
      {"StatelessRandomUniformInt"},
      {"StatelessRandomUniformIntV2"},
      {"StatelessRandomUniformV2"},
      {"StatelessTruncatedNormal"},
      {"StatelessTruncatedNormalV2"},
      {"StopGradient"},
      {"StridedSliceGrad", 2, {0, 4}},
      {"StringSplit"},
      {"StringToHashBucket"},
      {"StringToHashBucketFast"},
      {"StringToHashBucketStrong"},
      {"StringToNumber"},
      {"TFRecordReader"},
      {"Tanh"},
      {"TensorArray"},
      {"TensorArrayClose"},
      {"TensorArrayCloseV2"},
      {"TensorArrayCloseV3"},
      {"TensorArrayGrad"},
      {"TensorArrayGradV2"},
      {"TensorArrayGradV3"},
      {"TensorArrayGradWithShape"},
      {"TensorArrayScatter", 2, {2, 3}},
      {"TensorArrayScatterV2", 2, {2, 3}},
      {"TensorArrayScatterV3", 2, {2, 3}},
      {"TensorArraySize"},
      {"TensorArraySizeV2"},
      {"TensorArraySizeV3"},
      {"TensorArraySplit", 3, {1, 2, 3}},
      {"TensorArraySplitV2", 3, {1, 2, 3}},
      {"TensorArraySplitV3", 3, {1, 2, 3}},
      {"TensorArrayV2"},
      {"TensorArrayV3"},
      {"TensorArrayWrite", 2, {2, 3}},
      {"TensorArrayWriteV2", 2, {2, 3}},
      {"TensorArrayWriteV3", 2, {2, 3}},
      {"TensorListConcatLists"},
      {"TensorListConcatV2", 2, {1, 2}},
      {"TensorListElementShape"},
      {"TensorListFromTensor", 1, {1}},
      {"TensorListGetItem", 1, {2}},
      {"TensorListLength"},
      {"TensorListPopBack"},
      {"TensorListPushBack", 1, {0}},
      {"TensorListPushBackBatch"},
      {"TensorListScatter", 1, {2}},
      {"TensorListScatterV2", 2, {2, 3}},
      {"TensorListStack"},
      {"TensorScatterAdd", 2, {0, 2}},
      {"TensorScatterSub", 2, {0, 2}},
      {"TensorScatterUpdate", 1, {0}},
      {"TensorStridedSliceUpdate", 2, {0, 4}},
      {"TensorSummary"},
      {"TensorSummaryV2"},
      {"TextLineReader"},
      {"Timestamp"},
      {"TopKV2", 1, {1}},
      {"Transpose", 1, {0}},
      {"TridiagonalSolve", 1, {1}},
      {"TruncateDiv"},
      {"TruncatedNormal"},
      {"Unpack"},
      {"UnsortedSegmentSum", 2, {0, 2}},
      {"VarIsInitializedOp"},
      {"VariableShape"},
      {"WholeFileReader"},
      {"XlaClusterOutput"},
      {"XlaSharding"},
      {"XlaSpmdShardToFullShape"},
      {"ZerosLike"},
      {"_EagerConst"},
      {"VarHandleOp"},
  }};
  static const auto &m = *OpGradientInfoInit(a);

  auto it = m.find(op_name);
  if (it != m.end()) {
    return it->second;
  }
  return absl::nullopt;
}

absl::optional<tensorflow::gtl::FlatSet<int>> OpGradientUnusedOutputIndices(
    const tensorflow::string &op_name) {
  static std::array<OpIndexInfo, 481> a = {{
      {"Abs"},
      {"AccumulateNV2"},
      {"Acos"},
      {"Add"},
      {"AddN"},
      {"AddV2"},
      {"AllToAll"},
      {"Angle"},
      {"ApproximateEqual"},
      {"ArgMax"},
      {"ArgMin"},
      {"AsString"},
      {"Asin"},
      {"Assert"},
      {"Assign"},
      {"AssignAdd"},
      {"AssignSub"},
      {"Atan"},
      {"Atan2"},
      {"Atanh"},
      {"AudioSummary"},
      {"AudioSummaryV2"},
      {"AvgPool"},
      {"AvgPool3D"},
      {"AvgPool3DGrad"},
      {"AvgPoolGrad"},
      {"BatchMatMul"},
      {"BatchMatMulV2"},
      {"BatchMatMulV3"},
      {"BatchNormWithGlobalNormalization"},
      {"BatchToSpace"},
      {"BatchToSpaceND"},
      {"BesselI0"},
      {"BesselJ0"},
      {"BesselK0"},
      {"BesselY0"},
      {"Betainc"},
      {"BiasAdd"},
      {"BiasAddGrad"},
      {"BiasAddV1"},
      {"BitwiseAnd"},
      {"BitwiseOr"},
      {"BitwiseXor"},
      {"BroadcastGradientArgs"},
      {"BroadcastTo"},
      {"CSRSparseMatrixToDense"},
      {"CSRSparseMatrixToSparseTensor", 1, {1}},
      {"CTCGreedyDecoder"},
      {"CTCLoss", 1, {0}},
      {"CTCLossV2", 1, {0}},
      {"Cast"},
      {"Ceil"},
      {"CheckNumerics"},
      {"CheckNumericsV2"},
      {"CollectivePermute"},
      {"Complex"},
      {"CompositeTensorVariantFromComponents"},
      {"Concat"},
      {"ConcatV2"},
      {"Conj"},
      {"ConjugateTranspose"},
      {"Const"},
      {"Conv2D"},
      {"Conv2DBackpropFilter"},
      {"Conv2DBackpropInput"},
      {"Conv3D"},
      {"Conv3DBackpropFilterV2"},
      {"Conv3DBackpropInputV2"},
      {"Cos"},
      {"Cosh"},
      {"CropAndResize"},
      {"Cross"},
      {"CrossReplicaSum"},
      {"Cumprod"},
      {"Cumsum"},
      {"DebugGradientIdentity"},
      {"DebugGradientRefIdentity"},
      {"DebugIdentityV2"},
      {"DecodeBase64"},
      {"DecodePaddedRaw"},
      {"DecodeRaw"},
      {"DeleteSessionTensor"},
      {"DenseToCSRSparseMatrix"},
      {"DenseToDenseSetOperation"},
      {"DenseToSparseSetOperation"},
      {"DepthToSpace"},
      {"DepthwiseConv2dNative"},
      {"DepthwiseConv2dNativeBackpropFilter"},
      {"DepthwiseConv2dNativeBackpropInput"},
      {"Diag"},
      {"DiagPart"},
      {"Digamma"},
      {"Dilation2D"},
      {"Div"},
      {"DivNoNan"},
      {"DrawBoundingBoxes"},
      {"DynamicPartition"},
      {"EditDistance"},
      {"Einsum"},
      {"EluGrad"},
      {"EncodeBase64"},
      {"EncodeProto"},
      {"EnsureShape"},
      {"Enter"},
      {"Equal"},
      {"Erf"},
      {"Erfc"},
      {"Exit"},
      {"ExpandDims"},
      {"Expint"},
      {"Expm1"},
      {"ExtractGlimpse"},
      {"FFT"},
      {"FFT2D"},
      {"FFT3D"},
      {"FakeQuantWithMinMaxArgs"},
      {"FakeQuantWithMinMaxVars"},
      {"FakeQuantWithMinMaxVarsPerChannel"},
      {"Fill"},
      {"FixedLengthRecordReader"},
      {"Floor"},
      {"FloorDiv"},
      {"FloorMod"},
      {"FractionalAvgPool", 1, {0}},
      {"FresnelCos"},
      {"FresnelSin"},
      {"FusedBatchNorm", 3, {0, 1, 2}},
      {"FusedBatchNormGrad"},
      {"FusedBatchNormGradV2"},
      {"FusedBatchNormGradV3"},
      {"FusedBatchNormV2", 3, {0, 1, 2}},
      {"FusedBatchNormV3", 3, {0, 1, 2}},
      {"Gather"},
      {"GatherNd"},
      {"GatherV2"},
      {"GenerateBoundingBoxProposals"},
      {"GenerateVocabRemapping"},
      {"GetSessionHandle"},
      {"GetSessionHandleV2"},
      {"GetSessionTensor"},
      {"Greater"},
      {"GreaterEqual"},
      {"HSVToRGB"},
      {"HashTable"},
      {"HashTableV2"},
      {"HistogramSummary"},
      {"IFFT"},
      {"IFFT2D"},
      {"IFFT3D"},
      {"IRFFT"},
      {"IRFFT2D"},
      {"Identity"},
      {"IdentityN"},
      {"IdentityReader"},
      {"Igamma"},
      {"Igammac"},
      {"Imag"},
      {"ImageProjectiveTransformV2"},
      {"ImageProjectiveTransformV3"},
      {"ImageSummary"},
      {"InitializeTable"},
      {"InitializeTableFromTextFile"},
      {"InitializeTableFromTextFileV2"},
      {"InitializeTableV2"},
      {"InvGrad"},
      {"Invert"},
      {"InvertPermutation"},
      {"IsotonicRegression", 1, {0}},
      {"L2Loss"},
      {"LMDBReader"},
      {"LeakyRelu"},
      {"LeakyReluGrad"},
      {"LeftShift"},
      {"Less"},
      {"LessEqual"},
      {"Lgamma"},
      {"LinSpace"},
      {"LoadAndRemapMatrix"},
      {"Log"},
      {"Log1p"},
      {"LogMatrixDeterminant", 1, {0}},
      {"LogicalAnd"},
      {"LogicalNot"},
      {"LogicalOr"},
      {"LookupTableFind"},
      {"LookupTableFindV2"},
      {"LookupTableInsert"},
      {"LookupTableInsertV2"},
      {"LookupTableSize"},
      {"LookupTableSizeV2"},
      {"LoopCond"},
      {"MatMul"},
      {"MatrixBandPart"},
      {"MatrixDiag"},
      {"MatrixDiagPart"},
      {"MatrixDiagPartV2"},
      {"MatrixDiagPartV3"},
      {"MatrixDiagV2"},
      {"MatrixDiagV3"},
      {"MatrixSetDiag"},
      {"MatrixSetDiagV2"},
      {"MatrixSetDiagV3"},
      {"MaxPool3DGrad"},
      {"MaxPool3DGradGrad"},
      {"MaxPoolGrad"},
      {"MaxPoolGradGrad"},
      {"MaxPoolGradV2"},
      {"MaxPoolWithArgmax", 1, {0}},
      {"Maximum"},
      {"Merge", 1, {0}},
      {"MergeSummary"},
      {"Minimum"},
      {"MirrorPad"},
      {"MirrorPadGrad"},
      {"Mul"},
      {"MulNoNan"},
      {"Multinomial"},
      {"MutableDenseHashTable"},
      {"MutableDenseHashTableV2"},
      {"MutableHashTable"},
      {"MutableHashTableOfTensors"},
      {"MutableHashTableOfTensorsV2"},
      {"MutableHashTableV2"},
      {"NcclAllReduce"},
      {"NcclBroadcast"},
      {"NcclReduce"},
      {"Neg"},
      {"NextAfter"},
      {"NextIteration"},
      {"NonMaxSuppression"},
      {"NonMaxSuppressionV2"},
      {"NonMaxSuppressionWithOverlaps"},
      {"NotEqual"},
      {"OneHot"},
      {"OnesLike"},
      {"OptionalFromValue"},
      {"OptionalGetValue"},
      {"Pack"},
      {"Pad"},
      {"PadV2"},
      {"ParameterizedTruncatedNormal"},
      {"ParseTensor"},
      {"PlaceholderWithDefault"},
      {"Polygamma"},
      {"PopulationCount"},
      {"PreventGradient"},
      {"Print"},
      {"Prod"},
      {"QuantizeAndDequantize"},
      {"QuantizeAndDequantizeV2"},
      {"QuantizeAndDequantizeV3"},
      {"QuantizeAndDequantizeV4"},
      {"QuantizeAndDequantizeV4Grad"},
      {"QueueClose"},
      {"QueueEnqueue"},
      {"QueueEnqueueMany"},
      {"QueueSize"},
      {"RFFT"},
      {"RFFT2D"},
      {"RaggedGather"},
      {"RaggedRange"},
      {"RaggedTensorToSparse"},
      {"RaggedTensorToTensor"},
      {"RaggedTensorToVariant"},
      {"RandomCrop"},
      {"RandomIndexShuffle"},
      {"RandomStandardNormal"},
      {"RandomUniform"},
      {"Range"},
      {"Rank"},
      {"ReadVariableOp"},
      {"ReaderNumRecordsProduced"},
      {"ReaderNumWorkUnitsCompleted"},
      {"ReaderRead"},
      {"ReaderReadUpTo"},
      {"ReaderReset"},
      {"ReaderRestoreState"},
      {"ReaderSerializeState"},
      {"Real"},
      {"RealDiv"},
      {"ReciprocalGrad"},
      {"ReduceJoin"},
      {"RefEnter"},
      {"RefExit"},
      {"RefIdentity"},
      {"RefMerge", 1, {0}},
      {"RefNextIteration"},
      {"RefSwitch"},
      {"RegexReplace"},
      {"Relu6Grad"},
      {"ReluGrad"},
      {"Reshape"},
      {"ResizeBicubic"},
      {"ResizeBilinear"},
      {"ResizeNearestNeighbor"},
      {"ResourceGather"},
      {"ResourceGatherNd"},
      {"Reverse"},
      {"ReverseSequence"},
      {"ReverseV2"},
      {"RightShift"},
      {"Rint"},
      {"Roll"},
      {"Round"},
      {"RsqrtGrad"},
      {"SampleDistortedBoundingBox"},
      {"SampleDistortedBoundingBoxV2"},
      {"ScalarSummary"},
      {"ScaleAndTranslate"},
      {"ScatterAdd"},
      {"ScatterDiv"},
      {"ScatterMul"},
      {"ScatterNd"},
      {"ScatterNdAdd"},
      {"ScatterNdNonAliasingAdd"},
      {"ScatterNdSub"},
      {"ScatterNdUpdate"},
      {"ScatterSub"},
      {"SdcaFprint"},
      {"SdcaShrinkL1"},
      {"SegmentMean"},
      {"SegmentSum"},
      {"Select"},
      {"SeluGrad"},
      {"SerializeTensor"},
      {"SetSize"},
      {"Shape"},
      {"SigmoidGrad"},
      {"Sign"},
      {"Sin"},
      {"Sinh"},
      {"Size"},
      {"SoftmaxCrossEntropyWithLogits", 1, {0}},
      {"Softplus"},
      {"SoftplusGrad"},
      {"Softsign"},
      {"SpaceToBatch"},
      {"SpaceToBatchND"},
      {"SpaceToDepth"},
      {"SparseAdd", 2, {1, 2}},
      {"SparseAddGrad"},
      {"SparseConcat"},
      {"SparseDenseCwiseAdd"},
      {"SparseDenseCwiseDiv"},
      {"SparseDenseCwiseMul"},
      {"SparseFillEmptyRows", 3, {0, 1, 2}},
      {"SparseMatMul"},
      {"SparseMatrixAdd"},
      {"SparseMatrixMatMul"},
      {"SparseMatrixMul"},
      {"SparseMatrixNNZ"},
      {"SparseMatrixSparseMatMul"},
      {"SparseMatrixTranspose"},
      {"SparseMatrixZeros"},
      {"SparseReduceSum"},
      {"SparseReorder"},
      {"SparseSegmentMean"},
      {"SparseSegmentMeanWithNumSegments"},
      {"SparseSegmentSqrtN"},
      {"SparseSegmentSqrtNWithNumSegments"},
      {"SparseSegmentSum"},
      {"SparseSegmentSumWithNumSegments"},
      {"SparseSlice", 2, {1, 2}},
      {"SparseSoftmaxCrossEntropyWithLogits", 1, {0}},
      {"SparseSparseMaximum"},
      {"SparseSparseMinimum"},
      {"SparseTensorDenseAdd"},
      {"SparseTensorDenseMatMul"},
      {"SparseTensorToCSRSparseMatrix"},
      {"SparseToDense"},
      {"SparseToSparseSetOperation"},
      {"Spence"},
      {"Split"},
      {"SplitV"},
      {"Square"},
      {"SquaredDifference"},
      {"Squeeze"},
      {"Stack"},
      {"StackClose"},
      {"StackPop"},
      {"StackPush"},
      {"StatelessMultinomial"},
      {"StatelessRandomBinomial"},
      {"StatelessRandomNormal"},
      {"StatelessRandomNormalV2"},
      {"StatelessRandomPoisson"},
      {"StatelessRandomUniform"},
      {"StatelessRandomUniformFullInt"},
      {"StatelessRandomUniformFullIntV2"},
      {"StatelessRandomUniformInt"},
      {"StatelessRandomUniformIntV2"},
      {"StatelessRandomUniformV2"},
      {"StatelessTruncatedNormal"},
      {"StatelessTruncatedNormalV2"},
      {"StopGradient"},
      {"StridedSlice"},
      {"StridedSliceGrad"},
      {"StringJoin"},
      {"StringSplit"},
      {"StringToHashBucket"},
      {"StringToHashBucketFast"},
      {"StringToHashBucketStrong"},
      {"StringToNumber"},
      {"Sub"},
      {"Sum"},
      {"Switch"},
      {"TFRecordReader"},
      {"TPUEmbeddingActivations"},
      {"TPUReplicatedInput"},
      {"Tan"},
      {"TanhGrad"},
      {"TensorArray"},
      {"TensorArrayClose"},
      {"TensorArrayCloseV2"},
      {"TensorArrayCloseV3"},
      {"TensorArrayConcat", 1, {0}},
      {"TensorArrayConcatV2", 1, {0}},
      {"TensorArrayConcatV3", 1, {0}},
      {"TensorArrayGather"},
      {"TensorArrayGatherV2"},
      {"TensorArrayGatherV3"},
      {"TensorArrayGrad"},
      {"TensorArrayGradV2"},
      {"TensorArrayGradV3"},
      {"TensorArrayGradWithShape"},
      {"TensorArrayRead"},
      {"TensorArrayReadV2"},
      {"TensorArrayReadV3"},
      {"TensorArraySize"},
      {"TensorArraySizeV2"},
      {"TensorArraySizeV3"},
      {"TensorArrayV2"},
      {"TensorArrayV3"},
      {"TensorListConcat", 1, {0}},
      {"TensorListConcatLists"},
      {"TensorListConcatV2", 1, {0}},
      {"TensorListElementShape"},
      {"TensorListGather"},
      {"TensorListGetItem"},
      {"TensorListLength"},
      {"TensorListPushBack"},
      {"TensorListPushBackBatch"},
      {"TensorListResize"},
      {"TensorListScatter"},
      {"TensorListScatterIntoExistingList"},
      {"TensorListScatterV2"},
      {"TensorListSetItem"},
      {"TensorListSplit"},
      {"TensorListStack"},
      {"TensorScatterAdd"},
      {"TensorScatterSub"},
      {"TensorScatterUpdate"},
      {"TensorStridedSliceUpdate"},
      {"TensorSummary"},
      {"TensorSummaryV2"},
      {"TextLineReader"},
      {"Tile"},
      {"Timestamp"},
      {"TopK", 1, {0}},
      {"TopKV2", 1, {0}},
      {"Transpose"},
      {"TridiagonalMatMul"},
      {"TruncateDiv"},
      {"TruncatedNormal"},
      {"Unpack"},
      {"UnsortedSegmentSum"},
      {"VarIsInitializedOp"},
      {"VariableShape"},
      {"WholeFileReader"},
      {"Xdivy"},
      {"XlaClusterOutput"},
      {"XlaEinsum"},
      {"XlaSharding"},
      {"XlaSpmdFullToShardShape"},
      {"XlaSpmdShardToFullShape"},
      {"Xlog1py"},
      {"Xlogy"},
      {"ZerosLike"},
      {"Zeta"},
      {"_EagerConst"},
      {"VarHandleOp"},
  }};
  static const auto &m = *OpGradientInfoInit(a);

  auto it = m.find(op_name);
  if (it != m.end()) {
    return it->second;
  }
  return absl::nullopt;
}
