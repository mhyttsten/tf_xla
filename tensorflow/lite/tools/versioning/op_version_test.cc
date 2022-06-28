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
class MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_version_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_version_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_version_testDTcc() {
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
#include "tensorflow/lite/tools/versioning/op_version.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

// Creates vector of OpSignatureTensorSpec with the given TfLiteType vector.
std::vector<OpSignatureTensorSpec> CreateOpSignatureTensorSpecs(
    const std::vector<TfLiteType>& types) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  for (auto type : types) {
    OpSignatureTensorSpec tensor_spec = {};
    tensor_spec.type = type;
    tensor_specs.push_back(tensor_spec);
  }
  return tensor_specs;
}

// Creates vector of OpSignatureTensorSpec of single tensor spec of TfLiteType.
std::vector<OpSignatureTensorSpec> CreateOpSignatureTensorSpecs(
    const TfLiteType type) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  OpSignatureTensorSpec tensor_spec = {};
  tensor_spec.type = type;
  tensor_specs.push_back(tensor_spec);
  return tensor_specs;
}

// Creates vector of OpSignatureTensorSpec of single tensor spec of TfLiteType
// with shapes.
std::vector<OpSignatureTensorSpec> CreateOpSignatureTensorSpecs(
    const TfLiteType type, const int dim) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  OpSignatureTensorSpec tensor_spec = {};
  tensor_spec.type = type;
  for (int i = 0; i < dim; i++) {
    tensor_spec.dims.push_back(4);
  }
  tensor_specs.push_back(tensor_spec);
  return tensor_specs;
}

// Creates vector of OpSignatureTensorSpec of two tensor specs of TfLiteType
// with shapes.
std::vector<OpSignatureTensorSpec> CreateOpSignatureTensorSpecs(
    const TfLiteType type, const int dim1, const int dim2) {
  std::vector<OpSignatureTensorSpec> tensor_specs;
  OpSignatureTensorSpec tensor_spec1 = {};
  tensor_spec1.type = type;
  for (int i = 0; i < dim1; i++) {
    tensor_spec1.dims.push_back(4);
  }
  tensor_specs.push_back(tensor_spec1);

  OpSignatureTensorSpec tensor_spec2 = {};
  tensor_spec2.type = type;
  for (int i = 0; i < dim2; i++) {
    tensor_spec2.dims.push_back(4);
  }
  tensor_specs.push_back(tensor_spec2);
  return tensor_specs;
}

}  // namespace

TEST(OpVersionTest, VersioningSpareToDense) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8, kTfLiteInt8}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteUInt8, kTfLiteUInt8, kTfLiteUInt8}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt64, kTfLiteInt64, kTfLiteInt64}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_SPARSE_TO_DENSE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt32, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

// Test version for a simple Op with 2 versions and the input type controls the
// version.
void SimpleVersioningTest(BuiltinOperator op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_version_testDTcc mht_0(mht_0_v, 287, "", "./tensorflow/lite/tools/versioning/op_version_test.cc", "SimpleVersioningTest");

  OpSignature fake_op_sig = {
      .op = op,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = op,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

// Similar to SimpleVersioningTest function, but
// op has 3 versions and the input type includes kTfLiteInt16.
void SimpleVersioningTestExtended(BuiltinOperator op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_version_testDTcc mht_1(mht_1_v, 306, "", "./tensorflow/lite/tools/versioning/op_version_test.cc", "SimpleVersioningTestExtended");

  OpSignature fake_op_sig = {
      .op = op,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  SimpleVersioningTest(op);
}

// Test version for a simple Op with 2 versions and the output type controls the
void SimpleOutputVersioningTest(BuiltinOperator op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_version_testDTcc mht_2(mht_2_v, 320, "", "./tensorflow/lite/tools/versioning/op_version_test.cc", "SimpleOutputVersioningTest");

  OpSignature fake_op_sig = {
      .op = op,
      .inputs = std::vector<OpSignatureTensorSpec>{},
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = op,
      .inputs = std::vector<OpSignatureTensorSpec>{},
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningEqualTest) {
  SimpleVersioningTest(BuiltinOperator_EQUAL);
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_EQUAL,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteString),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningNotEqualTest) {
  SimpleVersioningTest(BuiltinOperator_NOT_EQUAL);
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_NOT_EQUAL,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteString),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningLessTest) {
  SimpleVersioningTest(BuiltinOperator_LESS);
}

TEST(OpVersionTest, VersioningLessEqualTest) {
  SimpleVersioningTest(BuiltinOperator_LESS_EQUAL);
}

TEST(OpVersionTest, VersioningGreaterTest) {
  SimpleVersioningTest(BuiltinOperator_GREATER);
}

TEST(OpVersionTest, VersioningGreaterEqualTest) {
  SimpleVersioningTest(BuiltinOperator_GREATER_EQUAL);
}

TEST(OpVersionTest, VersioningSpaceToBatchNDTest) {
  SimpleVersioningTest(BuiltinOperator_NOT_EQUAL);
}

TEST(OpVersionTest, VersioningLogSoftmaxTest) {
  SimpleVersioningTest(BuiltinOperator_LOG_SOFTMAX);
}

TEST(OpVersionTest, VersioningPackTest) {
  SimpleVersioningTest(BuiltinOperator_PACK);
}

TEST(OpVersionTest, VersioningUnpackTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_UNPACK,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_UNPACK,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_UNPACK,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningReluTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_RELU,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningBatchToSpaceNDTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_BATCH_TO_SPACE_ND,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 3);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_BATCH_TO_SPACE_ND,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 3);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningTanhTest) {
  SimpleVersioningTest(BuiltinOperator_TANH);
}

TEST(OpVersionTest, VersioningStridedSliceTest) {
  TfLiteStridedSliceParams strided_slice_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_STRIDED_SLICE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&strided_slice_params),
  };
  strided_slice_params.ellipsis_mask = 0;
  strided_slice_params.new_axis_mask = 2;
  fake_op_sig.ext_options.strided_slice.num_dims = 5;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  strided_slice_params.new_axis_mask = 0;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig.ext_options.strided_slice.num_dims = 4;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningSpaceToDepthTest) {
  SimpleVersioningTest(BuiltinOperator_SPACE_TO_DEPTH);
}

TEST(OpVersionTest, VersioningSliceTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SLICE,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteString, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_SLICE,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningLogisticTest) {
  SimpleVersioningTest(BuiltinOperator_SPACE_TO_DEPTH);
}

TEST(OpVersionTest, VersioningL2NormTest) {
  SimpleOutputVersioningTest(BuiltinOperator_L2_NORMALIZATION);
}

TEST(OpVersionTest, VersioningMaxTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_MAXIMUM,
  };

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 5, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_MAXIMUM,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningMinTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_MINIMUM,
  };

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 5, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_MINIMUM,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningMeanTest) {
  SimpleVersioningTestExtended(BuiltinOperator_MEAN);
}

TEST(OpVersionTest, VersioningSumTest) {
  SimpleVersioningTest(BuiltinOperator_SUM);
}

TEST(OpVersionTest, VersioningReduceMinTest) {
  SimpleVersioningTestExtended(BuiltinOperator_REDUCE_MIN);
}

TEST(OpVersionTest, VersioningReduceMaxTest) {
  SimpleVersioningTestExtended(BuiltinOperator_REDUCE_MAX);
}

TEST(OpVersionTest, VersioningReduceProdTest) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_REDUCE_PROD;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningAddTest) {
  TfLiteAddParams add_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_ADD,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&add_params)};
  add_params.pot_scale_int16 = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  SimpleVersioningTest(BuiltinOperator_ADD);
}

TEST(OpVersionTest, VersioningSubTest) {
  TfLiteSubParams sub_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SUB,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&sub_params)};
  sub_params.pot_scale_int16 = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt64);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8, 4, 5);
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  SimpleVersioningTest(BuiltinOperator_SUB);
}

TEST(OpVersionTest, VersioningMUL5Test) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_MUL;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt64);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);
}

TEST(OpVersionTest, VersioningSub4Test) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SUB,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt64),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}

void SimpleMulVersioningTest(TfLiteType data_type, float multiplier,
                             int version) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSversioningPSop_version_testDTcc mht_3(mht_3_v, 642, "", "./tensorflow/lite/tools/versioning/op_version_test.cc", "SimpleMulVersioningTest");

  OpSignature fake_op_sig = {
      .op = BuiltinOperator_MUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{data_type, data_type}),
      .outputs = CreateOpSignatureTensorSpecs(data_type),
  };
  fake_op_sig.ext_options.mul = {1.0f, 1.0f, 1.0f / multiplier};
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), version);
}

TEST(OpVersionTest, VersioningMulTest) {
  SimpleMulVersioningTest(kTfLiteUInt8, 0.5f, 1);
  SimpleMulVersioningTest(kTfLiteInt8, 0.5f, 2);
  SimpleMulVersioningTest(kTfLiteInt8, 2.0f, 3);
}

TEST(OpVersionTest, VersioningPadTest) {
  SimpleVersioningTest(BuiltinOperator_PAD);
}

TEST(OpVersionTest, VersioningPadV2Test) {
  SimpleVersioningTest(BuiltinOperator_PADV2);
}

TEST(OpVersionTest, VersioningConcatenationTest) {
  SimpleVersioningTest(BuiltinOperator_CONCATENATION);
}

TEST(OpVersionTest, VersioningSelectTest) {
  SimpleVersioningTest(BuiltinOperator_SELECT);
}

TEST(OpVersionTest, VersioningRelu6Test) {
  SimpleVersioningTestExtended(BuiltinOperator_RELU6);
}

TEST(OpVersionTest, VersioningFullyConnectedTest) {
  TfLiteFullyConnectedParams fully_connected_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteUInt8, kTfLiteUInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fully_connected_params.weights_format =
      kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fully_connected_params.weights_format =
      kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fully_connected_params.weights_format =
      kTfLiteFullyConnectedWeightsFormatDefault;
  fake_op_sig.ext_options.fully_connected.sparse_weight = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 8);

  fake_op_sig = {
      .op = BuiltinOperator_FULLY_CONNECTED,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&fully_connected_params),
  };
  fully_connected_params.asymmetric_quantize_inputs = false;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fully_connected_params.asymmetric_quantize_inputs = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 9);
}

TEST(OpVersionTest, VersioningDequantizeTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.ext_options.dequantize.is_per_channel_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig = {
      .op = BuiltinOperator_DEQUANTIZE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}

TEST(OpVersionTest, VersioningQuantizeTest) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_QUANTIZE;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  fake_op_sig.ext_options.quantize.is_per_channel_quantized = false;

  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.ext_options.quantize.is_per_channel_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningConv2DTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteUInt8, kTfLiteUInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  fake_op_sig.ext_options.conv_2d.is_per_channel_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig.op = BuiltinOperator_CONV_2D;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8});
  fake_op_sig.outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  fake_op_sig.ext_options.conv_2d.is_grouped_convolution = true;

  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);
}

TEST(OpVersionTest, VersioningFloorDivOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_FLOOR_DIV,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_FLOOR_DIV,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}

TEST(OpVersionTest, VersioningTransposeConvOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteUInt8}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt8, kTfLiteInt8}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(std::vector<TfLiteType>{
          kTfLiteInt32, kTfLiteInt8, kTfLiteInt8, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  const auto none_type = kTfLiteNoType;
  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE_CONV,
      .inputs = CreateOpSignatureTensorSpecs(std::vector<TfLiteType>{
          kTfLiteInt32, kTfLiteInt8, kTfLiteInt8, none_type}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}

TEST(OpVersionTest, VersioningSVDFOperatorTest) {
  TfLiteSVDFParams svdf_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SVDF,
      .inputs = CreateOpSignatureTensorSpecs(std::vector<TfLiteType>{
          kTfLiteFloat32, kTfLiteFloat32, kTfLiteFloat32, kTfLiteFloat32,
          kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&svdf_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_SVDF,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8, kTfLiteFloat32,
                                  kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&svdf_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  svdf_params.asymmetric_quantize_inputs = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);

  svdf_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_SVDF,
      .inputs = CreateOpSignatureTensorSpecs(std::vector<TfLiteType>{
          kTfLiteInt8, kTfLiteInt8, kTfLiteInt32, kTfLiteInt32, kTfLiteInt16}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&svdf_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningDepthwiseConv2DTest) {
  TfLiteDepthwiseConvParams depthwise_conv_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&depthwise_conv_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.ext_options.depthwise_conv_2d.is_per_channel_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 6);

  depthwise_conv_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&depthwise_conv_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&depthwise_conv_params),
  };
  depthwise_conv_params.dilation_width_factor = 2;
  depthwise_conv_params.dilation_height_factor = 2;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_DEPTHWISE_CONV_2D,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&depthwise_conv_params),
  };
  depthwise_conv_params.dilation_width_factor = 1;
  depthwise_conv_params.dilation_height_factor = 1;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTest, VersioningTileOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_TILE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_TILE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteString),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
TEST(OpVersionTest, VersioningTransposeTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 5);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteBool, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteBool, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_TRANSPOSE,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTest, VersioningGatherNdOperatorTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteString, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig = {
      .op = BuiltinOperator_GATHER_ND,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt32}),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}
TEST(OpVersionTest, VersioningDivTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_DIV,
  };
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 5, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 5, 5);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8, 4, 4);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTEst, VersioningFillTest) {
  OpSignature fake_op_sig = {BuiltinOperator_FILL};
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt8});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt64, kTfLiteInt16});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteBool});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteString});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(
      std::vector<TfLiteType>{kTfLiteInt32, kTfLiteInt32});
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);
}
TEST(OpVersionTest, VersioningResizeBilinearTest) {
  // Default.
  TfLiteResizeBilinearParams resize_bilinear_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RESIZE_BILINEAR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&resize_bilinear_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // align_corners=true is still version 1.
  resize_bilinear_params.align_corners = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // half_pixel_centers=true must be version 3.
  resize_bilinear_params.align_corners = false;
  resize_bilinear_params.half_pixel_centers = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int8 input is version 2.
  resize_bilinear_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_BILINEAR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&resize_bilinear_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  resize_bilinear_params.half_pixel_centers = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int16 input is version 4.
  resize_bilinear_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_BILINEAR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&resize_bilinear_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}
TEST(OpVersionTest, VersioningResizeNearestNeighborTest) {
  // Default.
  TfLiteResizeNearestNeighborParams resize_nearest_neighbor_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&resize_nearest_neighbor_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // align_corners=true is version 3.
  resize_nearest_neighbor_params.align_corners = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // half_pixel_centers=true must be version 3.
  resize_nearest_neighbor_params.align_corners = false;
  resize_nearest_neighbor_params.half_pixel_centers = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int8 input is version 2.
  resize_nearest_neighbor_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&resize_nearest_neighbor_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  resize_nearest_neighbor_params.align_corners = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // int16 input is version 4.
  resize_nearest_neighbor_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&resize_nearest_neighbor_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}
TEST(OpVersionTest, VersioningAbsTest) {
  // Default.
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_ABS,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // int8 input is version 2.
  fake_op_sig = {
      .op = BuiltinOperator_ABS,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  // int16 quantized input is version 3.
  fake_op_sig = {
      .op = BuiltinOperator_ABS,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  fake_op_sig.ext_options.abs.input_quantized = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
  // int16 non-quantized input is version 4.
  fake_op_sig = {
      .op = BuiltinOperator_ABS,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}
TEST(OpVersionTest, VersioningBatchMatMulTest) {
  // Default.
  TfLiteBatchMatMulParams batch_mat_mul_params = {};
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // int8 input is version 2.
  batch_mat_mul_params = {};
  fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  // int16 input is version 3.
  fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt16, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  // Symmetric hybrid quantized input is version 1.
  fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // Asymmetric hybrid quantized input is version 4.
  fake_op_sig = {
      .op = BuiltinOperator_BATCH_MATMUL,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .builtin_data = reinterpret_cast<void*>(&batch_mat_mul_params),
  };
  batch_mat_mul_params.asymmetric_quantize_inputs = true;
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 4);
}
TEST(OpVersionTest, VersioningSquaredDifferenceTest) {
  // Default.
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_SQUARED_DIFFERENCE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteFloat32, kTfLiteFloat32}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // int8 input is version 2.
  fake_op_sig = {
      .op = BuiltinOperator_SQUARED_DIFFERENCE,
      .inputs = CreateOpSignatureTensorSpecs(
          std::vector<TfLiteType>{kTfLiteInt8, kTfLiteInt8}),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
TEST(OpVersionTest, VersioningRsqrtTest) {
  // Default.
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_RSQRT,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  // int8 input is version 2.
  fake_op_sig = {
      .op = BuiltinOperator_RSQRT,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
TEST(OpVersionTest, VersioningBroadcastToTest) {
  OpSignature fake_op_sig = {
      .op = BuiltinOperator_BROADCAST_TO,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  // Quantized broadcast_to op is version 3.
  fake_op_sig = {
      .op = BuiltinOperator_BROADCAST_TO,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt8),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);

  fake_op_sig = {
      .op = BuiltinOperator_BROADCAST_TO,
      .inputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
      .outputs = CreateOpSignatureTensorSpecs(kTfLiteInt16),
  };
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 3);
}

TEST(OpVersionTest, VersioningGeluTest) {
  OpSignature fake_op_sig;
  fake_op_sig.op = BuiltinOperator_GELU;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteFloat32);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 1);

  fake_op_sig.op = BuiltinOperator_GELU;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);

  fake_op_sig.op = BuiltinOperator_GELU;
  fake_op_sig.inputs = CreateOpSignatureTensorSpecs(kTfLiteUInt8);
  EXPECT_EQ(GetBuiltinOperatorVersion(fake_op_sig), 2);
}
}  // namespace tflite
