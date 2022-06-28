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
class MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc() {
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
#include "tensorflow/lite/tools/optimize/calibration/logging_op_resolver.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace optimize {
namespace calibration {
namespace {

TfLiteStatus ConvPrepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc mht_0(mht_0_v, 195, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver_test.cc", "ConvPrepare");

  return kTfLiteOk;
}

TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc mht_1(mht_1_v, 202, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver_test.cc", "ConvEval");

  return kTfLiteOk;
}

TfLiteStatus AddPrepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc mht_2(mht_2_v, 209, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver_test.cc", "AddPrepare");

  return kTfLiteOk;
}

TfLiteStatus AddEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc mht_3(mht_3_v, 216, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver_test.cc", "AddEval");

  return kTfLiteOk;
}

TfLiteStatus CustomPrepare(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc mht_4(mht_4_v, 223, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver_test.cc", "CustomPrepare");

  return kTfLiteOk;
}

TfLiteStatus CustomEval(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc mht_5(mht_5_v, 230, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver_test.cc", "CustomEval");

  return kTfLiteOk;
}

TfLiteStatus WrappingInvoke(TfLiteContext* context, TfLiteNode* node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePStoolsPSoptimizePScalibrationPSlogging_op_resolver_testDTcc mht_6(mht_6_v, 237, "", "./tensorflow/lite/tools/optimize/calibration/logging_op_resolver_test.cc", "WrappingInvoke");

  return kTfLiteOk;
}

TEST(LoggingOpResolverTest, KernelInvokesAreReplaced) {
  MutableOpResolver base_resolver;
  TfLiteRegistration conv_registration = {};
  conv_registration.prepare = ConvPrepare;
  conv_registration.invoke = ConvEval;

  base_resolver.AddBuiltin(BuiltinOperator_CONV_2D, &conv_registration);

  TfLiteRegistration add_registration = {};
  add_registration.prepare = AddPrepare;
  add_registration.invoke = AddEval;

  base_resolver.AddBuiltin(BuiltinOperator_ADD, &add_registration);
  BuiltinOpsSet ops_to_replace = {
      {BuiltinOperator_CONV_2D, /*version*/ 1},
      {BuiltinOperator_ADD, /*version*/ 1},
  };

  LoggingOpResolver resolver(ops_to_replace, CustomOpsSet(), base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);

  auto reg = resolver.FindOp(BuiltinOperator_CONV_2D, 1);

  EXPECT_EQ(reg->builtin_code, BuiltinOperator_CONV_2D);
  EXPECT_TRUE(reg->prepare == ConvPrepare);
  EXPECT_TRUE(reg->invoke == WrappingInvoke);

  reg = resolver.FindOp(BuiltinOperator_ADD, 1);

  EXPECT_EQ(reg->builtin_code, BuiltinOperator_ADD);
  EXPECT_TRUE(reg->prepare == AddPrepare);
  EXPECT_TRUE(reg->invoke == WrappingInvoke);
}

TEST(LoggingOpResolverTest, OriginalKernelInvokesAreRetained) {
  MutableOpResolver base_resolver;
  TfLiteRegistration conv_registration = {};
  conv_registration.prepare = ConvPrepare;
  conv_registration.invoke = ConvEval;

  base_resolver.AddBuiltin(BuiltinOperator_CONV_2D, &conv_registration);

  TfLiteRegistration add_registration = {};
  add_registration.prepare = AddPrepare;
  add_registration.invoke = AddEval;

  base_resolver.AddBuiltin(BuiltinOperator_ADD, &add_registration);
  BuiltinOpsSet ops_to_replace = {
      {BuiltinOperator_CONV_2D, /*version*/ 1},
      {BuiltinOperator_ADD, /*version*/ 1},
  };

  LoggingOpResolver resolver(ops_to_replace, CustomOpsSet(), base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);
  auto kernel_invoke =
      resolver.GetWrappedKernelInvoke(BuiltinOperator_CONV_2D, 1);
  EXPECT_TRUE(kernel_invoke == ConvEval);
  kernel_invoke = resolver.GetWrappedKernelInvoke(BuiltinOperator_ADD, 1);
  EXPECT_TRUE(kernel_invoke == AddEval);
}

TEST(LoggingOpResolverTest, OnlyOpsInReplacementSetAreReplaces) {
  MutableOpResolver base_resolver;
  TfLiteRegistration conv_registration = {};
  conv_registration.prepare = ConvPrepare;
  conv_registration.invoke = ConvEval;

  base_resolver.AddBuiltin(BuiltinOperator_CONV_2D, &conv_registration);

  TfLiteRegistration add_registration = {};
  add_registration.prepare = AddPrepare;
  add_registration.invoke = AddEval;

  base_resolver.AddBuiltin(BuiltinOperator_ADD, &add_registration);
  // Only replace conv2d
  BuiltinOpsSet ops_to_replace = {
      {BuiltinOperator_CONV_2D, /*version*/ 1},
  };

  LoggingOpResolver resolver(ops_to_replace, CustomOpsSet(), base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);
  auto reg = resolver.FindOp(BuiltinOperator_CONV_2D, 1);
  EXPECT_EQ(reg->builtin_code, BuiltinOperator_CONV_2D);
  EXPECT_TRUE(reg->prepare == ConvPrepare);
  EXPECT_TRUE(reg->invoke == WrappingInvoke);

  reg = resolver.FindOp(BuiltinOperator_ADD, 1);
  EXPECT_EQ(nullptr, reg);
}

TEST(LoggingOpResolverTest, CustomOps) {
  MutableOpResolver base_resolver;
  TfLiteRegistration custom_registration = {};
  custom_registration.prepare = CustomPrepare;
  custom_registration.invoke = CustomEval;

  std::string custom_op_name = "custom";
  base_resolver.AddCustom(custom_op_name.c_str(), &custom_registration);

  CustomOpsSet ops_to_replace = {
      {custom_op_name, /*version*/ 1},
  };

  LoggingOpResolver resolver(BuiltinOpsSet(), ops_to_replace, base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);

  auto reg = resolver.FindOp(custom_op_name.c_str(), 1);

  EXPECT_EQ(reg->builtin_code, BuiltinOperator_CUSTOM);
  EXPECT_EQ(reg->custom_name, custom_op_name.c_str());
  EXPECT_TRUE(reg->prepare == CustomPrepare);
  EXPECT_TRUE(reg->invoke == WrappingInvoke);
}

TEST(LoggingOpResolverTest, UnresolvedCustomOps) {
  // No custom op registration.
  MutableOpResolver base_resolver;

  std::string custom_op_name = "unresolved_custom_op";

  CustomOpsSet ops_to_replace = {
      {custom_op_name, /*version*/ 1},
  };

  // Expect no death.
  LoggingOpResolver(BuiltinOpsSet(), ops_to_replace, base_resolver,
                    WrappingInvoke, /*error_reporter=*/nullptr);
}

TEST(LoggingOpResolverTest, UnresolvedBuiltinOps) {
  // No builtin op registration.
  MutableOpResolver base_resolver;

  BuiltinOpsSet ops_to_replace = {
      {BuiltinOperator_CONV_2D, /*version*/ 1},
      {BuiltinOperator_ADD, /*version*/ 1},
  };

  // Expect no death.
  LoggingOpResolver resolver(ops_to_replace, CustomOpsSet(), base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);
}

TEST(LoggingOpResolverTest, FlexOps) {
  // No flex op registration.
  MutableOpResolver base_resolver;

  std::string custom_op_name = "FlexAdd";

  CustomOpsSet ops_to_replace = {
      {custom_op_name, /*version*/ 1},
  };

  LoggingOpResolver resolver(BuiltinOpsSet(), ops_to_replace, base_resolver,
                             WrappingInvoke, /*error_reporter=*/nullptr);

  auto reg = resolver.FindOp(custom_op_name.c_str(), 1);

  EXPECT_TRUE(!reg);
}

}  // namespace
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite
