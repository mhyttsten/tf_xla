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
class MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/function_testlib.h"

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace test {
namespace function {

typedef FunctionDefHelper FDH;

GraphDef GDef(gtl::ArraySlice<NodeDef> nodes,
              gtl::ArraySlice<FunctionDef> funcs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/framework/function_testlib.cc", "GDef");

  GraphDef g;
  VersionDef* versions = g.mutable_versions();
  versions->set_producer(TF_GRAPH_DEF_VERSION);
  versions->set_min_consumer(TF_GRAPH_DEF_VERSION_MIN_CONSUMER);
  for (const auto& n : nodes) {
    *(g.add_node()) = n;
  }
  auto lib = g.mutable_library();
  for (const auto& f : funcs) {
    *(lib->add_function()) = f;
  }
  return g;
}

// Helper to construct a NodeDef.
NodeDef NDef(StringPiece name, StringPiece op, gtl::ArraySlice<string> inputs,
             gtl::ArraySlice<std::pair<string, FDH::AttrValueWrapper>> attrs,
             const string& device) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("device: \"" + device + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/framework/function_testlib.cc", "NDef");

  NodeDef n;
  n.set_name(string(name));
  n.set_op(string(op));
  for (const auto& in : inputs) n.add_input(in);
  n.set_device(device);
  for (const auto& na : attrs)
    n.mutable_attr()->insert({na.first, na.second.proto});
  return n;
}

FunctionDef NonZero() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/framework/function_testlib.cc", "NonZero");

  return FDH::Define(
      // Name
      "NonZero",
      // Args
      {"x:T"},
      // Return values
      {"y:T"},
      // Attr def
      {"T:{float, double, int32, int64, string}"},
      // Nodes
      {
          {{"y"}, "Identity", {"x"}, {{"T", "$T"}}},
      });
}

FunctionDef IsZero() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_3(mht_3_v, 256, "", "./tensorflow/core/framework/function_testlib.cc", "IsZero");

  const Tensor kZero = test::AsScalar<int64_t>(0);
  return FDH::Define(
      // Name
      "IsZero",
      // Args
      {"x: T"},
      // Return values
      {"equal: bool"},
      // Attr def
      {"T:{float, double, int32, int64, string}"},
      {
          {{"zero"}, "Const", {}, {{"value", kZero}, {"dtype", DT_INT64}}},
          {{"cast"}, "Cast", {"zero"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"equal"}, "Equal", {"x", "cast"}, {{"T", "$T"}}},
      });
}

FunctionDef RandomUniform() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_4(mht_4_v, 277, "", "./tensorflow/core/framework/function_testlib.cc", "RandomUniform");

  const Tensor kZero = test::AsScalar<int64_t>(0);

  return FDH::Define(
      // Name
      "RandomUniformFn",
      // Args
      {"x: T"},
      // Return values
      {"random_uniform: int64"},
      // Attr def
      {"T:{float, double, int32, int64, string}"},
      // NodeDef
      {{{"random_uniform/shape"},
        "Const",
        {},
        {{"value", kZero}, {"dtype", DT_INT64}}},
       {{"random_uniform"},
        "RandomUniform",
        {"random_uniform/shape"},
        {{"T", DT_INT32},
         {"dtype", DT_FLOAT},
         {"seed", 87654321},
         {"seed2", 42}}}});
}

FunctionDef XTimesTwo() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_5(mht_5_v, 306, "", "./tensorflow/core/framework/function_testlib.cc", "XTimesTwo");

  const Tensor kTwo = test::AsScalar<int64_t>(2);
  return FDH::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
}

FunctionDef TwoDeviceMult() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_6(mht_6_v, 328, "", "./tensorflow/core/framework/function_testlib.cc", "TwoDeviceMult");

  const Tensor kTwo = test::AsScalar<int64_t>(2);
  const Tensor kThree = test::AsScalar<int64_t>(3);
  return FDH::Create(
      // Name
      "TwoDeviceMult",
      // Args
      {"x: T"},
      // Return values
      {"y_cpu: T", "y_gpu: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"num_2"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"num_3"}, "Const", {}, {{"value", kThree}, {"dtype", DT_INT64}}},
          {{"factor_2"},
           "Cast",
           {"num_2:output:0"},
           {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"factor_3"},
           "Cast",
           {"num_3:output:0"},
           {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y_cpu"},
           "Mul",
           {"x", "factor_2:y:0"},
           {{"T", "$T"}},
           {},
           "/device:CPU:0"},
          {{"y_gpu"},
           "Mul",
           {"x", "factor_3:y:0"},
           {{"T", "$T"}},
           {},
           "/device:GPU:0"},
      },
      {{"y_cpu", "y_cpu:z:0"}, {"y_gpu", "y_gpu:z:0"}});
}

FunctionDef TwoDeviceInputOutput() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_7(mht_7_v, 371, "", "./tensorflow/core/framework/function_testlib.cc", "TwoDeviceInputOutput");

  const Tensor kTwo = test::AsScalar<float>(2);
  const Tensor kThree = test::AsScalar<float>(3);
  return FDH::Create(
      // Name
      "TwoDeviceInputOutput",
      // Args
      {"x1: T", "x2: T"},
      // Return values
      {"y_cpu: T", "y_gpu: T"},
      // Attr def
      {"T: {float}"},
      // Nodes
      {
          {{"num_2"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_FLOAT}}},
          {{"num_3"}, "Const", {}, {{"value", kThree}, {"dtype", DT_FLOAT}}},
          {{"y_cpu"},
           "Mul",
           {"x1", "num_2:output:0"},
           {{"T", "$T"}},
           {},
           "/device:CPU:0"},
          {{"y_gpu"},
           "Mul",
           {"x2", "num_3:output:0"},
           {{"T", "$T"}},
           {},
           "/device:GPU:0"},
      },
      {{"y_cpu", "y_cpu:z:0"}, {"y_gpu", "y_gpu:z:0"}});
}

FunctionDef FuncWithListInput() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_8(mht_8_v, 406, "", "./tensorflow/core/framework/function_testlib.cc", "FuncWithListInput");

  const Tensor kTwo = test::AsScalar<float>(2);
  return FDH::Create(
      // Name
      "FuncWithListInput",
      // Args
      {"x1: N * T"},
      // Return values
      {},
      // Attr def
      {"T: {float}", "N: int >= 1"},
      // Nodes
      {
          {{"num_2"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_FLOAT}}},
      },
      {});
}

FunctionDef FuncWithListOutput() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_9(mht_9_v, 427, "", "./tensorflow/core/framework/function_testlib.cc", "FuncWithListOutput");

  const Tensor kTwo = test::AsScalar<float>(2);
  return FDH::Create(
      // Name
      "FuncWithListOutput",
      // Args
      {},
      // Return values
      {"y: N * T"},
      // Attr def
      {"T: {float}", "N: int >= 1"},
      // Nodes
      {
          {{"num_2"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_FLOAT}}},
      },
      {{"y", "num_2:output:0"}});
}

FunctionDef XAddX() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_10(mht_10_v, 448, "", "./tensorflow/core/framework/function_testlib.cc", "XAddX");

  return FDH::Define(
      // Name
      "XAddX",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"y"}, "Add", {"x", "x"}, {{"T", "$T"}}},
      });
}

FunctionDef XAddY() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_11(mht_11_v, 467, "", "./tensorflow/core/framework/function_testlib.cc", "XAddY");

  return FDH::Define(
      // Name
      "XAddY",
      // Args
      {"x: T", "y: T"},
      // Return values
      {"z: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"z"}, "Add", {"x", "y"}, {{"T", "$T"}}},
      });
}

FunctionDef XTimesTwoInt32() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_12(mht_12_v, 486, "", "./tensorflow/core/framework/function_testlib.cc", "XTimesTwoInt32");

  const Tensor kTwo = test::AsScalar<int64_t>(2);
  return FDH::Define(
      // Name
      "XTimesTwoInt32",
      // Args
      {"x: int32"},
      // Return values
      {"y: int32"}, {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"},
           "Cast",
           {"two"},
           {{"SrcT", DT_INT64}, {"DstT", DT_INT32}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", DT_INT32}}},
      });
}

FunctionDef XTimesFour() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_13(mht_13_v, 509, "", "./tensorflow/core/framework/function_testlib.cc", "XTimesFour");

  return FDH::Create(
      // Name
      "XTimesFour",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"x2"}, "XTimesTwo", {"x"}, {{"T", "$T"}}},
          {{"y"}, "XTimesTwo", {"x2:y:0"}, {{"T", "$T"}}},
      },
      {{"y", "y:y:0"}});
}

FunctionDef XTimes16() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_14(mht_14_v, 530, "", "./tensorflow/core/framework/function_testlib.cc", "XTimes16");

  return FDH::Create(
      // Name
      "XTimes16",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"x4"}, "XTimesFour", {"x"}, {{"T", "$T"}}},
          {{"y"}, "XTimesFour", {"x4:y:0"}, {{"T", "$T"}}},
      },
      {{"y", "y:y:0"}});
}

FunctionDef WXPlusB() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_15(mht_15_v, 551, "", "./tensorflow/core/framework/function_testlib.cc", "WXPlusB");

  return FDH::Define(
      // Name
      "WXPlusB",
      // Args
      {"w: T", "x: T", "b: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double}"},
      // Nodes
      {{{"mm"},
        "MatMul",
        {"w", "x"},
        {{"T", "$T"}, {"transpose_a", false}, {"transpose_b", false}}},
       {{"y"}, "Add", {"mm", "b"}, {{"T", "$T"}}}});
}

FunctionDef Swap() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_16(mht_16_v, 572, "", "./tensorflow/core/framework/function_testlib.cc", "Swap");

  return FDH::Define(
      // Name
      "Swap",
      // Args
      {"i0: T", "i1: T"},
      // Return values
      {"o0: T", "o1: T"},
      // Attr def
      {"T: {float, double, resource}"},
      // Nodes
      {{{"o0"}, "Identity", {"i1"}, {{"T", "$T"}}},
       {{"o1"}, "Identity", {"i0"}, {{"T", "$T"}}}});
}

FunctionDef EmptyBodySwap() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_17(mht_17_v, 590, "", "./tensorflow/core/framework/function_testlib.cc", "EmptyBodySwap");

  return FDH::Create(
      // Name
      "EmptyBodySwap",
      // Args
      {"i0: T", "i1: T"},
      // Return values
      {"o0: T", "o1: T"},
      // Attr def
      {"T: {float, double, resource}"},
      // Nodes
      {},
      // Output mapping
      {{"o0", "i1"}, {"o1", "i0"}});
}

FunctionDef ResourceOutput() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_18(mht_18_v, 609, "", "./tensorflow/core/framework/function_testlib.cc", "ResourceOutput");

  const Tensor kTwo = test::AsScalar<float>(2);
  return FDH::Create(
      // Name
      "ResourceOutput",
      // Args
      {"x: float", "y: resource"},
      // Return values
      {"y_out: resource", "two_x: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_FLOAT}}},
          {{"mul"}, "Mul", {"x", "two:output:0"}, {{"T", DT_FLOAT}}, {}},
      },
      {{"y_out", "y"}, {"two_x", "mul:z:0"}});
}

FunctionDef ResourceIdentity() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_19(mht_19_v, 631, "", "./tensorflow/core/framework/function_testlib.cc", "ResourceIdentity");

  return FDH::Create(
      // Name
      "ResourceIdentity",
      // Args
      {"x: resource"},
      // Return values
      {"y: resource"},
      // Attr def
      {},
      // Nodes
      {},
      // Output mapping
      {{"y", "x"}});
}

FunctionDef ReadResourceVariable() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_20(mht_20_v, 650, "", "./tensorflow/core/framework/function_testlib.cc", "ReadResourceVariable");

  return FDH::Create(
      // Name
      "ReadResourceVariable",
      // Args
      {"x: resource"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"read"}, "ReadVariableOp", {"x"}, {{"dtype", DT_FLOAT}}, {}},
      },
      {{"y", "read:value:0"}});
}

FunctionDef ControlFlow() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_21(mht_21_v, 670, "", "./tensorflow/core/framework/function_testlib.cc", "ControlFlow");

  return FDH::Create(
      // Name
      "ControlFlow",
      // Args
      {"i: float"},
      // Return values
      {"o: float"},
      // Attr def
      {},
      // Nodes
      {{{"enter"}, "Enter", {"i"}, {{"T", DT_FLOAT}, {"frame_name", "while"}}}},
      // Output mapping
      {{"o", "enter:output"}});
}

FunctionDef InvalidControlFlow() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_22(mht_22_v, 689, "", "./tensorflow/core/framework/function_testlib.cc", "InvalidControlFlow");

  return FDH::Create(
      // Name
      "InvalidControlFlow",
      // Args
      {"i: int32"},
      // Return values
      {"o: int32"},
      // Attr def
      {},
      // Nodes
      {{{"enter"}, "Enter", {"i"}, {{"T", DT_INT32}, {"frame_name", "while"}}},
       {{"add"}, "Add", {"enter:output", "i"}, {{"T", DT_INT32}}}},
      // Output mapping
      {{"o", "add:z"}});
}

FunctionDef LessThanOrEqualToN(int64_t N) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_23(mht_23_v, 709, "", "./tensorflow/core/framework/function_testlib.cc", "LessThanOrEqualToN");

  const Tensor kN = test::AsScalar<int64_t>(N);
  return FDH::Define(
      // Name
      "LessThanOrEqualToN",
      // Args
      {"x: T"},
      // Return values
      {"z: bool"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"N"}, "Const", {}, {{"value", kN}, {"dtype", DT_INT64}}},
          {{"y"}, "Cast", {"N"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"z"}, "LessEqual", {"x", "y"}, {{"T", "$T"}}},
      });
}

FunctionDef XPlusOneXTimesY() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_24(mht_24_v, 731, "", "./tensorflow/core/framework/function_testlib.cc", "XPlusOneXTimesY");

  const Tensor kOne = test::AsScalar<int64_t>(1);
  return FDH::Define(
      // Name
      "XPlusOneXTimesY",
      // Args
      {"x: T", "y: T"},
      // Return values
      {"s: T", "t: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {{{"one"}, "Const", {}, {{"value", kOne}, {"dtype", DT_INT64}}},
       {{"increment"}, "Cast", {"one"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
       {{"s"}, "Add", {"x", "increment"}, {{"T", "$T"}}},
       {{"t"}, "Mul", {"x", "y"}, {{"T", "$T"}}}});
}

FunctionDef XYXLessThanOrEqualToN(int64_t N) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_25(mht_25_v, 752, "", "./tensorflow/core/framework/function_testlib.cc", "XYXLessThanOrEqualToN");

  const Tensor kN = test::AsScalar<int64_t>(N);
  return FDH::Define(
      // Name
      "XYXLessThanOrEqualToN",
      // Args
      {"x: T", "y: T"},
      // Return values
      {"z: bool"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"N"}, "Const", {}, {{"value", kN}, {"dtype", DT_INT64}}},
          {{"N1"}, "Cast", {"N"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"z"}, "LessEqual", {"x", "N1"}, {{"T", "$T"}}},
      });
}

FunctionDef RandomUniformLess() {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_26(mht_26_v, 774, "", "./tensorflow/core/framework/function_testlib.cc", "RandomUniformLess");

  const Tensor kZero = test::AsScalar<int32>(0);
  const Tensor kOne = test::AsScalar<int32>(1);
  const Tensor k005 = test::AsScalar<float>(0.05);

  return FDH::Define(
      // Name
      "RandomUniformLess",
      // Args
      {"arg0: int64"},
      // Return values
      {"strided_slice: bool"},
      // Attr def
      {"T:{float, double, int32, int64, string}"},
      {{{"random_uniform/shape"},
        "Const",
        {},
        {{"value", kZero}, {"dtype", DT_INT32}}},

       {{"random_uniform/RandomUniform"},
        "RandomUniform",
        {"random_uniform/shape"},
        {{"T", DT_INT32}, {"Tout", DT_FLOAT}, {"seed", 0}, {"seed2", 0}}},

       {{"Less/y"}, "Const", {}, {{"value", k005}, {"dtype", DT_FLOAT}}},

       {{"Less"},
        "Less",
        {"random_uniform/RandomUniform", "Less/y"},
        {{"T", DT_FLOAT}}},

       {{"strided_slice/stack"},
        "Const",
        {},
        {{"value", kZero}, {"dtype", DT_INT32}}},

       {{"strided_slice/stack_1"},
        "Const",
        {},
        {{"value", kOne}, {"dtype", DT_INT32}}},

       {{"strided_slice/stack_2"},
        "Const",
        {},
        {{"value", kOne}, {"dtype", DT_INT32}}},

       {{"strided_slice"},
        "StridedSlice",
        {"Less", "strided_slice/stack", "strided_slice/stack_1",
         "strided_slice/stack_2"},
        {{"Index", DT_INT32},
         {"T", DT_BOOL},
         {"begin_mask", 0},
         {"ellipsis_mask", 0},
         {"end_mask", 0},
         {"new_axis_mask", 0},
         {"shrink_axis_mask", 0}}}});
}

FunctionDef MakeRangeDataset() {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_27(mht_27_v, 836, "", "./tensorflow/core/framework/function_testlib.cc", "MakeRangeDataset");

  return FDH::Define(
      /*name=*/"MakeRangeDataset",
      /*arg_def=*/{"start: int64", "stop: int64", "step: int64"},
      /*ret_def=*/{"y:variant"},
      /*attr_def=*/
      {"output_types: list(type) >= 1", "output_shapes: list(shape) >= 1"},
      /*node_def=*/
      {{/*ret=*/{"y"},
        /*op=*/"RangeDataset",
        /*arg=*/{"start", "stop", "step"},
        /*attr=*/
        {{"output_types", "$output_types"},
         {"output_shapes", "$output_shapes"}}}});
}

FunctionDef MakeBatchDataset() {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_28(mht_28_v, 855, "", "./tensorflow/core/framework/function_testlib.cc", "MakeBatchDataset");

  return FDH::Define(
      /*name=*/"MakeBatchDataset",
      /*arg_def=*/
      {"input_dataset: variant", "batch_size: int64", "drop_remainder: bool"},
      /*ret_def=*/{"y: variant"},
      /*attr_def=*/
      {"parallel_copy: bool = false", "output_types: list(type) >= 1",
       "output_shapes: list(shape) >= 1"},
      /*node_def=*/
      {{/*ret=*/{"y"},
        /*op=*/"BatchDatasetV2",
        /*arg=*/{"input_dataset", "batch_size", "drop_remainder"},
        /*attr=*/
        {{"parallel_copy", "$parallel_copy"},
         {"output_types", "$output_types"},
         {"output_shapes", "$output_shapes"}}}});
}

FunctionDef MakeMapDataset(bool has_other_args) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_29(mht_29_v, 877, "", "./tensorflow/core/framework/function_testlib.cc", "MakeMapDataset");

  std::vector<string> args = {"input_dataset: variant"};
  std::vector<string> inputs = {"input_dataset"};
  if (has_other_args) {
    args.emplace_back("other_arguments: Targuments");
    inputs.emplace_back("other_arguments");
  }

  return FDH::Define(
      /*name=*/"MakeMapDataset",
      /*arg_def=*/args,
      /*ret_def=*/
      {"y: variant"},
      /*attr_def=*/
      {"f: func", "Targuments: list(type) >= 0",
       "output_types: list(type) >= 1", "output_shapes: list(shape) >= 1",
       "use_inter_op_parallelism: bool = true",
       "preserve_cardinality: bool = false"},
      /*node_def=*/
      {{/*ret=*/{"y"},
        /*op=*/"MapDataset",
        /*arg=*/inputs,
        /*attr=*/
        {{"f", "$f"},
         {"Targuments", "$Targuments"},
         {"output_types", "$output_types"},
         {"output_shapes", "$output_shapes"},
         {"use_inter_op_parallelism", "$use_inter_op_parallelism"},
         {"preserve_cardinality", "$preserve_cardinality"}}}});
}

FunctionDef MakeTakeDataset() {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_30(mht_30_v, 911, "", "./tensorflow/core/framework/function_testlib.cc", "MakeTakeDataset");

  return FDH::Define(
      // Name
      "TakeDataset",
      // Args
      {"input_dataset: variant", "count: int64"},
      // Return values
      {"y:variant"},
      // Attr def
      {"output_types: list(type) >= 1", "output_shapes: list(shape) >= 1"},
      // Nodes
      {{{"y"},
        "TakeDataset",
        {"input_dataset", "count"},
        {{"output_types", "$output_types"},
         {"output_shapes", "$output_shapes"}}}});
}

FunctionDef MakeTensorSliceDataset() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_31(mht_31_v, 932, "", "./tensorflow/core/framework/function_testlib.cc", "MakeTensorSliceDataset");

  return FDH::Define(
      // Name
      "MakeTensorSliceDataset",
      // Args
      {"x: Toutput_types"},
      // Return values
      {"y: variant"},
      // Attr def
      {"Toutput_types: list(type) >= 1", "output_shapes: list(shape) >= 1"},
      // Nodes
      {{{"y"},
        "TensorSliceDataset",
        {"x"},
        {{"Toutput_types", "$Toutput_types"},
         {"output_shapes", "$output_shapes"}}}});
}

FunctionDef Unique() {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_32(mht_32_v, 953, "", "./tensorflow/core/framework/function_testlib.cc", "Unique");

  return FDH::Create(
      // Name
      "GetUnique",
      // Args
      {"x:T"},
      // Return values
      {"y:T", "idx: out_idx"},
      // Attr def
      {"T: type", "out_idx: {int32, int64} = DT_INT32"},
      // Nodes
      {
          {{"result"}, "Unique", {"x"}, {{"T", "$T"}, {"out_idx", "$out_idx"}}},
      },
      {{"y", "result:y:0"}, {"idx", "result:idx:0"}});
}

void FunctionTestSchedClosure(std::function<void()> fn) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSfunction_testlibDTcc mht_33(mht_33_v, 973, "", "./tensorflow/core/framework/function_testlib.cc", "FunctionTestSchedClosure");

  static thread::ThreadPool* w =
      new thread::ThreadPool(Env::Default(), "Test", 8);
  w->Schedule(std::move(fn));
}

}  // end namespace function
}  // end namespace test
}  // end namespace tensorflow
