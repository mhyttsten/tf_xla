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
class MHTracer_DTPStensorflowPSlitePSkernelsPStable_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPStable_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPStable_testDTcc() {
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
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_TABLE();

namespace {

using ::testing::ElementsAreArray;

class TableOpModel : public SingleOpModel {
 public:
  TableOpModel(const TensorData& input, const TensorData& table,
               const TensorData& output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStable_testDTcc mht_0(mht_0_v, 204, "", "./tensorflow/lite/kernels/table_test.cc", "TableOpModel");

    input_ = AddInput(input);
    table_ = AddInput(table);
    output_ = AddOutput(output);
    SetCustomOp("Table", {}, Register_TABLE);
    BuildInterpreter({GetShape(input_), GetShape(table_)});
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename integer_dtype>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<integer_dtype>(ExtractVector<integer_dtype>(output_),
                                     GetScale(output_), GetZeroPoint(output_));
  }

  int input() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStable_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/kernels/table_test.cc", "input");
 return input_; }
  int table() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStable_testDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/kernels/table_test.cc", "table");
 return table_; }
  int output() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStable_testDTcc mht_3(mht_3_v, 234, "", "./tensorflow/lite/kernels/table_test.cc", "output");
 return output_; }

 protected:
  int input_;
  int table_;
  int output_;
};

// A LUT of 256 values is used in the int8 case. For the int16 case a 513 LUT is
// used but as the last value is only used for interpolation we only have 512
// quantized steps.
template <typename T>
inline float GetLUTTolerance(float input_min, float input_max, float output_min,
                             float output_max) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPStable_testDTcc mht_4(mht_4_v, 250, "", "./tensorflow/lite/kernels/table_test.cc", "GetLUTTolerance");

  static_assert(
      std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value,
      "T must be an int8_t or int16_t.");

  const float range_sum = (input_max - input_min) + (output_max - output_min);
  if (std::is_same<T, int8_t>::value) {
    return range_sum / 256.0f;
  } else {
    return range_sum / 512.0f;
  }
}

template <typename InputT, typename OutputT>
void TableWithExpLUTToInt8Test() {
  using TableT = OutputT;

  float input_min = -0.5f;
  float input_max = 0.8f;
  // Use symmetric inputs for int16 cases, nudge max for null zero-point
  if (std::is_same<InputT, int16_t>::value) {
    input_min = -0.8f;
    input_max = 0.8f * std::numeric_limits<InputT>::max() /
                static_cast<float>(std::numeric_limits<InputT>::max() + 1);
  }

  float output_min = 0.0f;
  float output_max = 2.4f;
  // Use symmetric outputs  for int16 cases, nudge max for null zero-point
  if (std::is_same<OutputT, int16_t>::value) {
    output_min = -2.4f;
    output_max = 2.4f * std::numeric_limits<OutputT>::max() /
                 static_cast<float>(std::numeric_limits<OutputT>::max() + 1);
  }

  const float kQuantizedTolerance =
      GetLUTTolerance<TableT>(input_min, input_max, output_min, output_max);

  std::vector<TableT> table(lut_size<InputT>());
  TableOpModel m({GetTensorType<InputT>(), {1, 2, 3, 1}, input_min, input_max},
                 {GetTensorType<TableT>(), {lut_size<InputT>()}},
                 {GetTensorType<OutputT>(), {}, output_min, output_max});

  // -1.204706 = m.GetScale(m.output()) * m.GetZeroPoint(m.output()). It's added
  // to avoid capture with function pointers.
  gen_lut<float, InputT, TableT>(
      [](float v) { return std::exp(v) - 1.204706f; }, input_min, input_max,
      output_min, output_max, table.data());

  m.QuantizeAndPopulate<InputT>(m.input(),
                                {-0.5f, -0.2f, 0.0f, 0.1f, 0.3f, 0.8f});
  m.PopulateTensor<TableT>(m.table(), table);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<OutputT>(),
              ElementsAreArray(ArrayFloatNear(
                  {std::exp(-0.5f), std::exp(-0.2f), std::exp(0.0f),
                   std::exp(0.1f), std::exp(0.3f), std::exp(0.8f)},
                  kQuantizedTolerance)));
}

template <typename InputT, typename OutputT>
void TableWithExpLUTToInt16Test() {
  using TableT = OutputT;

  float input_min = -0.5f;
  float input_max = 0.8f;
  // Use symmetric inputs for int16 cases, nudge max for null zero-point
  if (std::is_same<InputT, int16_t>::value) {
    input_min = -0.8f;
    input_max = 0.8f * std::numeric_limits<InputT>::max() /
                static_cast<float>(std::numeric_limits<InputT>::max() + 1);
  }

  float output_min = 0.0f;
  float output_max = 2.4f;
  // Use symmetric outputs  for int16 cases, nudge max for null zero-point
  if (std::is_same<OutputT, int16_t>::value) {
    output_min = -2.4f;
    output_max = 2.4f * std::numeric_limits<OutputT>::max() /
                 static_cast<float>(std::numeric_limits<OutputT>::max() + 1);
  }

  const float kQuantizedTolerance =
      GetLUTTolerance<TableT>(input_min, input_max, output_min, output_max);

  std::vector<TableT> table(lut_size<InputT>());
  TableOpModel m({GetTensorType<InputT>(), {1, 2, 3, 1}, input_min, input_max},
                 {GetTensorType<TableT>(), {lut_size<InputT>()}},
                 {GetTensorType<OutputT>(), {}, output_min, output_max});

  gen_lut<float, InputT, TableT>([](float v) { return std::exp(v); }, input_min,
                                 input_max, output_min, output_max,
                                 table.data());

  m.QuantizeAndPopulate<InputT>(m.input(),
                                {-0.5f, -0.2f, 0.0f, 0.1f, 0.3f, 0.8f});
  m.PopulateTensor<TableT>(m.table(), table);
  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(m.GetDequantizedOutput<OutputT>(),
              ElementsAreArray(ArrayFloatNear(
                  {std::exp(-0.5f), std::exp(-0.2f), std::exp(0.0f),
                   std::exp(0.1f), std::exp(0.3f), std::exp(0.8f)},
                  kQuantizedTolerance)));
}

TEST(TableOpTest, Int8ToInt8WithExpLUT) {
  TableWithExpLUTToInt8Test<int8_t, int8_t>();
}

TEST(TableOpTest, Int8ToInt16WithExpLUT) {
  TableWithExpLUTToInt16Test<int8_t, int16_t>();
}

TEST(TableOpTest, Int16ToInt16WithExpLUT) {
  TableWithExpLUTToInt16Test<int16_t, int16_t>();
}

TEST(TableOpTest, Int16ToInt8WithExpLUT) {
  TableWithExpLUTToInt8Test<int16_t, int8_t>();
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
