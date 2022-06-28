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
class MHTracer_DTPStensorflowPSlitePStocoPStoco_convert_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPStoco_convert_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPStoco_convert_testDTcc() {
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
#include "tensorflow/lite/toco/toco_convert.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/toco/toco_port.h"

namespace toco {
namespace {

TEST(TocoTest, MissingInputFile) {
  ParsedTocoFlags toco_flags;
  ParsedModelFlags model_flags;
  EXPECT_DEATH(Convert(toco_flags, model_flags).ok(),
               "Missing required flag --input_file");
}

TEST(TocoTest, BadInputFormat) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  std::string input;
  std::string output;

  EXPECT_DEATH(Convert(input, toco_flags, model_flags, &output).ok(),
               "Unhandled input_format='FILE_FORMAT_UNKNOWN'");
}

TEST(TocoTest, MissingOutputArrays) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);
  std::string input;
  std::string output;

  EXPECT_DEATH(Convert(input, toco_flags, model_flags, &output).ok(),
               "This model does not define output arrays, so a --output_arrays "
               "flag must be given on the command-line");
}

TEST(TocoTest, BadOutputArray) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);
  model_flags.add_output_arrays("output1");
  std::string input;
  std::string output;

  EXPECT_DEATH(Convert(input, toco_flags, model_flags, &output).ok(),
               "Specified output array .output1. is not produced by any op "
               "in this graph. Is it a typo");
}

TEST(TocoTest, BadOutputFormat) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);
  model_flags.add_output_arrays("output1");
  std::string input = R"GraphDef(
    node {
      name: "output1"
      input: "input1"
      input: "input2"
      op: "Sub"
      attr { key: "T" value { type: DT_FLOAT } }
    }
  )GraphDef";

  std::string output;

  EXPECT_DEATH(Convert(input, toco_flags, model_flags, &output).ok(),
               "Unhandled output_format='FILE_FORMAT_UNKNOWN'");
}

TEST(TocoTest, SimpleFloatModel) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);
  toco_flags.set_output_format(TENSORFLOW_GRAPHDEF);

  // Inputs are automatically selected (but that might not be a good idea).
  model_flags.add_output_arrays("output1");
  std::string input = R"GraphDef(
    node {
      name: "input1"
      op: "Placeholder"
      attr { key: "dtype" value { type: DT_INT64 } }
    }
    node {
      name: "input2"
      op: "Placeholder"
      attr { key: "dtype" value { type: DT_INT64 } }
    }
    node {
      name: "output1"
      input: "input1"
      input: "input2"
      op: "Sub"
      attr { key: "T" value { type: DT_FLOAT } }
    }
  )GraphDef";

  std::string output;
  EXPECT_TRUE(Convert(input, toco_flags, model_flags, &output).ok());
  EXPECT_TRUE(!output.empty());
}

TEST(TocoTest, TransientStringTensors) {
  TocoFlags toco_flags;
  ModelFlags model_flags;

  toco_flags.set_input_format(TENSORFLOW_GRAPHDEF);

  // We need to do a couple of things to trigger the transient array
  // initialization code: output format must support memory planning, and the
  // input array must have a shape.
  toco_flags.set_output_format(TFLITE);

  toco::InputArray* input_1 = model_flags.add_input_arrays();
  input_1->set_name("input1");
  toco::InputArray* indices_1 = model_flags.add_input_arrays();
  indices_1->set_name("indices1");

  model_flags.add_output_arrays("output1");
  std::string input = R"GraphDef(
    node {
      name: "input1"
      op: "Placeholder"
      attr { key: "dtype" value { type: DT_STRING } }
      attr { key: "shape" value { shape { dim { size:1 }}}}
    }
    node {
      name: "indices1"
      op: "Placeholder"
      attr { key: "dtype" value { type: DT_INT64 } }
    }
    node {
      name: "intermediate1"
      op: "Gather"
      input: "input1"
      input: "indices1"
      attr { key: "Tparams" value { type: DT_STRING } }
      attr { key: "Tindices" value { type: DT_INT64 } }
    }
    node {
      name: "output1"
      op: "Gather"
      input: "intermediate1"
      input: "indices2"
      attr { key: "Tparams" value { type: DT_STRING } }
      attr { key: "Tindices" value { type: DT_INT64 } }
    }
  )GraphDef";

  std::string output;

  EXPECT_TRUE(Convert(input, toco_flags, model_flags, &output).ok());
  EXPECT_TRUE(!output.empty());
}

}  // namespace
}  // namespace toco

int main(int argc, char** argv) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPStoco_convert_testDTcc mht_0(mht_0_v, 350, "", "./tensorflow/lite/toco/toco_convert_test.cc", "main");

  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  ::toco::port::InitGoogleWasDoneElsewhere();
  return RUN_ALL_TESTS();
}
