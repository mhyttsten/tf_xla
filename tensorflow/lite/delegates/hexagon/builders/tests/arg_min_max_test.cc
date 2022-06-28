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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarg_min_max_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarg_min_max_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarg_min_max_testDTcc() {
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
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/builders/tests/hexagon_delegate_op_model.h"

namespace tflite {
using testing::ElementsAreArray;

class ArgBaseOpModel : public SingleOpModelWithHexagon {
 public:
  explicit ArgBaseOpModel(TensorType input_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarg_min_max_testDTcc mht_0(mht_0_v, 193, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arg_min_max_test.cc", "ArgBaseOpModel");

    input_ = AddInput(input_type);
    output_ = AddOutput(TensorType_INT32);
  }

  int input() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarg_min_max_testDTcc mht_1(mht_1_v, 201, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arg_min_max_test.cc", "input");
 return input_; }

  std::vector<int> GetInt32Output() const {
    return ExtractVector<int>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  using SingleOpModelWithHexagon::builder_;

  int input_;
  int output_;
};

class ArgMinOpModel : public ArgBaseOpModel {
 public:
  ArgMinOpModel(std::initializer_list<int> input_shape, TensorType input_type)
      : ArgBaseOpModel(input_type /*input_type*/), input_shape_(input_shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarg_min_max_testDTcc mht_2(mht_2_v, 221, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arg_min_max_test.cc", "ArgMinOpModel");
}

  void Build() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarg_min_max_testDTcc mht_3(mht_3_v, 226, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arg_min_max_test.cc", "Build");

    SetBuiltinOp(BuiltinOperator_ARG_MIN, BuiltinOptions_ArgMinOptions,
                 CreateArgMinOptions(builder_, TensorType_INT32 /*output_type*/)
                     .Union());
    BuildInterpreter({input_shape_, {1}});
  }

 private:
  std::vector<int> input_shape_;
};

class ArgMaxOpModel : public ArgBaseOpModel {
 public:
  ArgMaxOpModel(std::initializer_list<int> input_shape, TensorType input_type)
      : ArgBaseOpModel(input_type /*input_type*/), input_shape_(input_shape) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarg_min_max_testDTcc mht_4(mht_4_v, 243, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arg_min_max_test.cc", "ArgMaxOpModel");
}

  void Build() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStestsPSarg_min_max_testDTcc mht_5(mht_5_v, 248, "", "./tensorflow/lite/delegates/hexagon/builders/tests/arg_min_max_test.cc", "Build");

    SetBuiltinOp(BuiltinOperator_ARG_MAX, BuiltinOptions_ArgMaxOptions,
                 CreateArgMaxOptions(builder_, TensorType_INT32 /*output_type*/)
                     .Union());
    BuildInterpreter({input_shape_, {1}});
  }

 private:
  std::vector<int> input_shape_;
};

template <typename integer_type, TensorType tensor_dtype>
void ArgMinTestImpl() {
  ArgMinOpModel model({1, 1, 1, 4}, tensor_dtype);
  model.AddConstInput(TensorType_INT32, {3}, {1});
  model.Build();

  if (tensor_dtype == TensorType_UINT8) {
    model.SymmetricQuantizeAndPopulate(model.input(), {1, 5, 0, 7});
  } else {
    model.SignedSymmetricQuantizeAndPopulate(model.input(), {1, 5, 0, 7});
  }
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetInt32Output(), ElementsAreArray({2}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1}));
}

template <typename integer_type, TensorType tensor_dtype>
void ArgMinNegativeTestImpl() {
  ArgMinOpModel model({1, 1, 2, 4}, tensor_dtype);
  model.AddConstInput(TensorType_INT32, {-2}, {1});
  model.Build();

  if (tensor_dtype == TensorType_UINT8) {
    model.SymmetricQuantizeAndPopulate(model.input(), {1, 2, 7, 8, 1, 9, 7, 3});
  } else {
    model.SignedSymmetricQuantizeAndPopulate(model.input(),
                                             {1, 2, 7, 8, 1, 9, 7, 3});
  }
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetInt32Output(), ElementsAreArray({0, 0, 0, 1}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 4}));
}

template <typename integer_type, TensorType tensor_dtype>
void ArgMaxTestImpl() {
  ArgMaxOpModel model({1, 1, 1, 4}, tensor_dtype);
  model.AddConstInput(TensorType_INT32, {3}, {1});
  model.Build();

  if (tensor_dtype == TensorType_UINT8) {
    model.SymmetricQuantizeAndPopulate(model.input(), {1, 5, 0, 7});
  } else {
    model.SignedSymmetricQuantizeAndPopulate(model.input(), {1, 5, 0, 7});
  }
  model.ApplyDelegateAndInvoke();
  EXPECT_THAT(model.GetInt32Output(), ElementsAreArray({3}));
}

TEST(ArgMinTest, GetArgMin_UInt8) {
  ArgMinTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ArgMinTest, GetArgMin_Int8) { ArgMinTestImpl<int8_t, TensorType_INT8>(); }

TEST(ArgMinTest, GetArgMinNegative_UInt8) {
  ArgMinNegativeTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ArgMinTest, GetArgMinNegative_Int8) {
  ArgMinNegativeTestImpl<int8_t, TensorType_INT8>();
}

TEST(ArgMaxTest, GetArgMax_UInt8) {
  ArgMaxTestImpl<uint8_t, TensorType_UINT8>();
}

TEST(ArgMaxTest, GetArgMax_Int8) { ArgMaxTestImpl<int8_t, TensorType_INT8>(); }
}  // namespace tflite
