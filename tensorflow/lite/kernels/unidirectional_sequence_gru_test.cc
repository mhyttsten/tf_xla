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
class MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc() {
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

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_UNIDIRECTIONAL_SEQUENCE_GRU();

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class GRUOpModel : public SingleOpModel {
 public:
  explicit GRUOpModel(int n_batch, int n_input, int n_output,
                      const std::vector<std::vector<int>>& input_shapes,
                      const TensorType& weight_type = TensorType_FLOAT32)
      : n_batch_(n_batch), n_input_(n_input), n_output_(n_output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "GRUOpModel");

    input_ = AddInput(TensorType_FLOAT32);
    input_state_ =
        AddVariableInput(TensorData{TensorType_FLOAT32, {n_batch, n_output}});
    gate_weight_ = AddInput(TensorType_FLOAT32);
    gate_bias_ = AddInput(TensorType_FLOAT32);
    candidate_weight_ = AddInput(TensorType_FLOAT32);
    candidate_bias_ = AddInput(TensorType_FLOAT32);

    output_ = AddOutput(TensorType_FLOAT32);
    output_state_ = AddOutput(TensorType_FLOAT32);

    SetCustomOp("UNIDIRECTIONAL_SEQUENCE_GRU", {},
                Register_UNIDIRECTIONAL_SEQUENCE_GRU);
    BuildInterpreter(input_shapes);
  }

  void SetInput(const std::vector<float>& f) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "SetInput");
 PopulateTensor(input_, f); }

  void SetInputState(const std::vector<float>& f) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_2(mht_2_v, 231, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "SetInputState");

    PopulateTensor(input_state_, f);
  }

  void SetGateWeight(const std::vector<float>& f) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_3(mht_3_v, 238, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "SetGateWeight");

    PopulateTensor(gate_weight_, f);
  }

  void SetGateBias(const std::vector<float>& f) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_4(mht_4_v, 245, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "SetGateBias");

    PopulateTensor(gate_bias_, f);
  }

  void SetCandidateWeight(const std::vector<float>& f) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_5(mht_5_v, 252, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "SetCandidateWeight");

    PopulateTensor(candidate_weight_, f);
  }

  void SetCandidateBias(const std::vector<float>& f) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_6(mht_6_v, 259, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "SetCandidateBias");

    PopulateTensor(candidate_bias_, f);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  int num_batches() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_7(mht_7_v, 270, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "num_batches");
 return n_batch_; }
  int num_inputs() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_8(mht_8_v, 274, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "num_inputs");
 return n_input_; }
  int num_outputs() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSunidirectional_sequence_gru_testDTcc mht_9(mht_9_v, 278, "", "./tensorflow/lite/kernels/unidirectional_sequence_gru_test.cc", "num_outputs");
 return n_output_; }

 private:
  int input_;
  int input_state_;
  int gate_weight_;
  int gate_bias_;
  int candidate_weight_;
  int candidate_bias_;

  int output_;
  int output_state_;
  int n_batch_;
  int n_input_;
  int n_output_;
};

TEST(GRUTest, SimpleTest) {
  const int n_time = 2;
  const int n_batch = 2;
  const int n_input = 2;
  const int n_output = 3;

  GRUOpModel m(n_batch, n_input, n_output,
               {{n_time, n_batch, n_input},
                {n_batch, n_output},
                {2 * n_output, n_input + n_output},
                {2 * n_output},
                {n_output, n_input + n_output},
                {n_output}});
  // All data is randomly generated.
  m.SetInput({0.89495724, 0.34482682, 0.68505806, 0.7135783, 0.3167085,
              0.93647677, 0.47361764, 0.39643127});
  m.SetInputState(
      {0.09992421, 0.3028481, 0.78305984, 0.50438094, 0.11269058, 0.10244724});
  m.SetGateWeight({0.7256918,  0.8945897,  0.03285786, 0.42637166, 0.119376324,
                   0.83035135, 0.16997327, 0.42302176, 0.77598256, 0.2660894,
                   0.9587266,  0.6218451,  0.88164485, 0.12272458, 0.2699055,
                   0.18399088, 0.21930052, 0.3374841,  0.70866305, 0.9523419,
                   0.25170696, 0.60988617, 0.79823977, 0.64477515, 0.2602957,
                   0.5053131,  0.93722224, 0.8451359,  0.97905475, 0.38669217});
  m.SetGateBias(
      {0.032708533, 0.018445263, 0.15320699, 0.8163046, 0.26683575, 0.1412022});
  m.SetCandidateWeight({0.96165305, 0.95572084, 0.11534478, 0.96965164,
                        0.33562955, 0.8680755, 0.003066936, 0.057793964,
                        0.8671354, 0.33354893, 0.7313398, 0.78492093,
                        0.19530584, 0.116550304, 0.13599132});
  m.SetCandidateBias({0.89837056, 0.54769796, 0.63364106});

  ASSERT_EQ(m.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputShape(), ElementsAre(n_time, n_batch, n_output));
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {0.20112592, 0.45286041, 0.80842507, 0.59567153, 0.2619998,
                   0.22922856, 0.27715868, 0.5247152, 0.82300174, 0.65812796,
                   0.38217607, 0.3401444})));
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
