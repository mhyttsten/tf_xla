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
class MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc() {
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

#include "tensorflow/core/grappler/optimizers/function_api_info.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {
void SetArg(const string& name, const string& type_name,
            OpDef::ArgDef* arg_def) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("type_name: \"" + type_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/grappler/optimizers/function_api_info_test.cc", "SetArg");

  arg_def->set_name(name);
  arg_def->set_type_attr(type_name);
}

typedef std::pair<string, string> ArgSpec;  // name, type.

void SetArgs(const std::vector<ArgSpec>& input_args_spec,
             const std::vector<ArgSpec>& output_args_spec, OpDef* sig) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/grappler/optimizers/function_api_info_test.cc", "SetArgs");

  for (const auto& arg_spec : input_args_spec)
    SetArg(arg_spec.first, arg_spec.second, sig->add_input_arg());
  for (const auto& arg_spec : output_args_spec)
    SetArg(arg_spec.first, arg_spec.second, sig->add_output_arg());
}

void PopulateFunction(const string& name, const string& api_interface_name,
                      const string& preferred_device,
                      const std::vector<ArgSpec>& input_args,
                      const std::vector<ArgSpec>& output_args,
                      const string& forward_function_name,
                      const string& backward_function_name,
                      FunctionDef* func_def) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("name: \"" + name + "\"");
   mht_2_v.push_back("api_interface_name: \"" + api_interface_name + "\"");
   mht_2_v.push_back("preferred_device: \"" + preferred_device + "\"");
   mht_2_v.push_back("forward_function_name: \"" + forward_function_name + "\"");
   mht_2_v.push_back("backward_function_name: \"" + backward_function_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc mht_2(mht_2_v, 235, "", "./tensorflow/core/grappler/optimizers/function_api_info_test.cc", "PopulateFunction");

  OpDef* sig = func_def->mutable_signature();
  sig->set_name(name);

  SetArgs(input_args, output_args, sig);

  auto* func_attr = func_def->mutable_attr();
  if (!api_interface_name.empty())
    (*func_attr)["api_implements"].set_s(api_interface_name);
  if (!preferred_device.empty())
    (*func_attr)["api_preferred_device"].set_s(preferred_device);
  if (!forward_function_name.empty())
    (*func_attr)["forward_function_name"].set_s(forward_function_name);
  if (!backward_function_name.empty())
    (*func_attr)["backward_function_name"].set_s(backward_function_name);
}

void PopulateSampleLibrary(const bool mismatch_args,
                           FunctionDefLibrary* func_lib) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc mht_3(mht_3_v, 256, "", "./tensorflow/core/grappler/optimizers/function_api_info_test.cc", "PopulateSampleLibrary");

  const std::vector<ArgSpec> func_args{{"in1", "float32"}, {"in2", "int32"}};
  const std::vector<ArgSpec> func_wrong_args{{"in1", "int32"},
                                             {"in2", "int32"}};
  const std::vector<ArgSpec> output_args{{"out", "float32"}};
  PopulateFunction("DoStuffCpu", "DoStuff", "CPU", func_args, output_args, "",
                   "", func_lib->add_function());
  PopulateFunction("DoStuffGpu", "DoStuff", "GPU",
                   mismatch_args ? func_wrong_args : func_args, output_args, "",
                   "", func_lib->add_function());
  PopulateFunction("DoThings", "DoThings", "", func_args, output_args, "", "",
                   func_lib->add_function());
  PopulateFunction("OneOff", "", "", func_args, output_args, "", "",
                   func_lib->add_function());
  PopulateFunction("AnotherOneOff", "", "", func_args, output_args, "", "",
                   func_lib->add_function());
}

void PopulateComplexLibrary(FunctionDefLibrary* func_lib) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc mht_4(mht_4_v, 277, "", "./tensorflow/core/grappler/optimizers/function_api_info_test.cc", "PopulateComplexLibrary");

  const std::vector<ArgSpec> input_args{{"in1", "float32"}, {"in2", "int32"}};
  const std::vector<ArgSpec> output_args{{"out", "float32"}};
  const std::vector<ArgSpec> output_with_state{
      {"out", "float32"}, {"state1", "int32"}, {"state2", "int32"}};

  PopulateFunction("DoStuffCpu", "DoStuff", "CPU", input_args, output_args, "",
                   "DoStuffCpu_gradient", func_lib->add_function());
  PopulateFunction("DoStuffCpu_gradient", "DoStuff", "CPU", output_args,
                   input_args, "DoStuffCpu", "", func_lib->add_function());
  PopulateFunction("DoStuffGpu", "DoStuff", "GPU", input_args,
                   output_with_state, "", "DoStuffGpu_gradient",
                   func_lib->add_function());
  PopulateFunction("DoStuffGpu_gradient", "DoStuff", "GPU", output_with_state,
                   input_args, "DoStuffGpu", "", func_lib->add_function());
}

bool CheckEquivImpl(const FunctionLibraryApiInfo& lib_api_info,
                    const string& func_name,
                    const std::vector<string>& expected_other) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc mht_5(mht_5_v, 300, "", "./tensorflow/core/grappler/optimizers/function_api_info_test.cc", "CheckEquivImpl");

  std::vector<string> other_impl;
  Status status =
      lib_api_info.GetEquivalentImplementations(func_name, &other_impl);
  EXPECT_EQ(status, Status::OK());
  const std::unordered_set<string> actual(other_impl.begin(), other_impl.end());
  const std::unordered_set<string> expected(expected_other.begin(),
                                            expected_other.end());
  return actual == expected;
}

string GetInterfaceName(const FunctionLibraryApiInfo& lib_api_info,
                        const string& func_name) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc mht_6(mht_6_v, 316, "", "./tensorflow/core/grappler/optimizers/function_api_info_test.cc", "GetInterfaceName");

  auto* info = lib_api_info.GetApiInfo(func_name);
  CHECK_NOTNULL(info);
  return info->interface_name();
}

string GetPreferredDevice(const FunctionLibraryApiInfo& lib_api_info,
                          const string& func_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("func_name: \"" + func_name + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSoptimizersPSfunction_api_info_testDTcc mht_7(mht_7_v, 327, "", "./tensorflow/core/grappler/optimizers/function_api_info_test.cc", "GetPreferredDevice");

  auto* info = lib_api_info.GetApiInfo(func_name);
  CHECK_NOTNULL(info);
  return info->preferred_device();
}

TEST(FunctionApiInfoTest, ParseTags) {
  FunctionDefLibrary func_lib;
  PopulateSampleLibrary(/* mismatch_args */ false, &func_lib);
  FunctionLibraryApiInfo lib_api_info;
  TF_ASSERT_OK(lib_api_info.Init(func_lib));

  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffCpu"));
  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffGpu"));
  EXPECT_EQ("DoThings", GetInterfaceName(lib_api_info, "DoThings"));

  EXPECT_EQ("CPU", GetPreferredDevice(lib_api_info, "DoStuffCpu"));
  EXPECT_EQ("GPU", GetPreferredDevice(lib_api_info, "DoStuffGpu"));
  EXPECT_EQ("", GetPreferredDevice(lib_api_info, "DoThings"));

  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffCpu", {"DoStuffGpu"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffGpu", {"DoStuffCpu"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "Undefined", {}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "OneOff", {}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "AnotherOneOff", {}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoThings", {}));
}

TEST(FunctionApiInfoTest, ComplexFunctionLib) {
  FunctionDefLibrary func_lib;
  PopulateComplexLibrary(&func_lib);
  FunctionLibraryApiInfo lib_api_info;
  TF_ASSERT_OK(lib_api_info.Init(func_lib));

  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffCpu"));
  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffCpu_gradient"));
  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffGpu"));
  EXPECT_EQ("DoStuff", GetInterfaceName(lib_api_info, "DoStuffGpu_gradient"));

  EXPECT_EQ("CPU", GetPreferredDevice(lib_api_info, "DoStuffCpu"));
  EXPECT_EQ("CPU", GetPreferredDevice(lib_api_info, "DoStuffCpu_gradient"));
  EXPECT_EQ("GPU", GetPreferredDevice(lib_api_info, "DoStuffGpu"));
  EXPECT_EQ("GPU", GetPreferredDevice(lib_api_info, "DoStuffGpu_gradient"));

  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffCpu", {"DoStuffGpu"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffGpu", {"DoStuffCpu"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffCpu_gradient",
                             {"DoStuffGpu_gradient"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "DoStuffGpu_gradient",
                             {"DoStuffCpu_gradient"}));
  EXPECT_TRUE(CheckEquivImpl(lib_api_info, "Undefined", {}));
}

TEST(FunctionApiInfoTest, MismatchedArguments) {
  FunctionDefLibrary func_lib;
  PopulateSampleLibrary(/* mismatch_args */ true, &func_lib);
  FunctionLibraryApiInfo lib_api_info;
  const Status ret = lib_api_info.Init(func_lib);
  EXPECT_FALSE(ret.ok());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
