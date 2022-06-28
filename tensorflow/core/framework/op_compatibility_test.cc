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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc() {
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

// Test that verifies that various changes to an OpDef are
// backwards-compatible.

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class TestKernel : public OpKernel {
 public:
  explicit TestKernel(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/framework/op_compatibility_test.cc", "TestKernel");
}
  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/framework/op_compatibility_test.cc", "Compute");

    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("ndef", TensorShape({}),
                                                     &out_tensor));
    out_tensor->scalar<tstring>()() = SummarizeNodeDef(def());
  }
};

class OpCompatibilityTest : public OpsTestBase {
 protected:
  const OpDef* RegisteredOpDef() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/framework/op_compatibility_test.cc", "RegisteredOpDef");

    const OpDef* op_def;
    TF_CHECK_OK(OpRegistry::Global()->LookUpOpDef(node_def()->op(), &op_def));
    return op_def;
  }

  void ExpectSuccess(const OpDef& old_op_def) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_3(mht_3_v, 229, "", "./tensorflow/core/framework/op_compatibility_test.cc", "ExpectSuccess");

    // Record the original signature before we change *node_def().
    DataTypeVector old_in_types, old_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), old_op_def, &old_in_types,
                                   &old_out_types));

    // This should be all that is needed to get compatibility.
    const OpDef* new_op_def = RegisteredOpDef();
    AddDefaultsToNodeDef(*new_op_def, node_def());

    // Validate that it is indeed compatible.
    TF_ASSERT_OK(ValidateNodeDef(*node_def(), *new_op_def));
    DataTypeVector new_in_types, new_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), *new_op_def, &new_in_types,
                                   &new_out_types));
    if (new_in_types.size() == old_in_types.size()) {
      // Ref inputs are allowed to become non-ref inputs.
      for (int i = 0; i < new_in_types.size(); ++i) {
        if (IsRefType(old_in_types[i]) && !IsRefType(new_in_types[i])) {
          old_in_types[i] = RemoveRefType(old_in_types[i]);
        }
      }
    }
    ASSERT_EQ(new_in_types, old_in_types);
    if (new_out_types.size() == old_out_types.size()) {
      // Non-ref outputs are allowed to become ref outputs.
      for (int i = 0; i < new_out_types.size(); ++i) {
        if (!IsRefType(old_out_types[i]) && IsRefType(new_out_types[i])) {
          old_out_types[i] = MakeRefType(old_out_types[i]);
        }
      }
    }
    ASSERT_EQ(new_out_types, old_out_types);
    TF_ASSERT_OK(OpDefCompatible(old_op_def, *new_op_def));

    // Verify the Op actually runs.  Result() will return the output.
    TF_ASSERT_OK(InitOp());
    TF_ASSERT_OK(RunOpKernel());
  }

  string Result() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_4(mht_4_v, 272, "", "./tensorflow/core/framework/op_compatibility_test.cc", "Result");
 return GetOutput(0)->scalar<tstring>()(); }

  void ExpectIncompatible(const OpDef& old_op_def, const OpDef& new_op_def,
                          const string& error) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("error: \"" + error + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_5(mht_5_v, 279, "", "./tensorflow/core/framework/op_compatibility_test.cc", "ExpectIncompatible");

    // Test OpDefCompatible gives the same answer without the node_def.
    Status status = OpDefCompatible(old_op_def, new_op_def);
    if (status.ok()) {
      ADD_FAILURE() << SummarizeOpDef(old_op_def) << " vs. "
                    << SummarizeOpDef(new_op_def);
    } else {
      EXPECT_TRUE(absl::StrContains(status.error_message(), error))
          << status << " does not contain " << error;
    }
  }

  void ExpectInvalid(const OpDef& old_op_def, const string& validation_error,
                     const string& compatibility_error) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("validation_error: \"" + validation_error + "\"");
   mht_6_v.push_back("compatibility_error: \"" + compatibility_error + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_6(mht_6_v, 297, "", "./tensorflow/core/framework/op_compatibility_test.cc", "ExpectInvalid");

    // Record the original signature before we change *node_def().
    DataTypeVector old_in_types, old_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), old_op_def, &old_in_types,
                                   &old_out_types));

    // This should be all that is needed to get compatibility.
    const OpDef* new_op_def = RegisteredOpDef();
    AddDefaultsToNodeDef(*new_op_def, node_def());

    // Validate that it does not pass validation.
    Status status = ValidateNodeDef(*node_def(), *new_op_def);
    if (status.ok()) {
      ADD_FAILURE() << SummarizeNodeDef(*node_def());
    } else {
      EXPECT_TRUE(absl::StrContains(status.error_message(), validation_error))
          << status << " does not contain " << validation_error;
    }

    ExpectIncompatible(old_op_def, *new_op_def, compatibility_error);
  }

  void ExpectTypeMismatch(const OpDef& old_op_def,
                          const string& compatibility_error) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("compatibility_error: \"" + compatibility_error + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_7(mht_7_v, 324, "", "./tensorflow/core/framework/op_compatibility_test.cc", "ExpectTypeMismatch");

    // Record the original signature before we change *node_def().
    DataTypeVector old_in_types, old_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), old_op_def, &old_in_types,
                                   &old_out_types));

    // This should be all that is needed to get compatibility.
    const OpDef* new_op_def = RegisteredOpDef();
    AddDefaultsToNodeDef(*new_op_def, node_def());

    // Validate that it is valid, but with incompatible types.
    TF_ASSERT_OK(ValidateNodeDef(*node_def(), *new_op_def));

    DataTypeVector new_in_types, new_out_types;
    TF_ASSERT_OK(InOutTypesForNode(*node_def(), *new_op_def, &new_in_types,
                                   &new_out_types));
    if (new_in_types == old_in_types && new_out_types == old_out_types) {
      ADD_FAILURE() << SummarizeNodeDef(*node_def()) << "\n"
                    << DataTypeVectorString(new_in_types) << " -> "
                    << DataTypeVectorString(new_out_types);
    }

    ExpectIncompatible(old_op_def, *new_op_def, compatibility_error);
  }

  void ExpectRenameFailure(const OpDef& old_op_def,
                           const string& compatibility_error) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("compatibility_error: \"" + compatibility_error + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_8(mht_8_v, 354, "", "./tensorflow/core/framework/op_compatibility_test.cc", "ExpectRenameFailure");

    // This should be all that is needed to get compatibility.
    const OpDef* new_op_def = RegisteredOpDef();
    AddDefaultsToNodeDef(*new_op_def, node_def());

    // Validate that the NodeDef is valid.  This will ignore
    // problems caused by output name changes for functions.
    TF_ASSERT_OK(ValidateNodeDef(*node_def(), *new_op_def));

    ExpectIncompatible(old_op_def, *new_op_def, compatibility_error);
  }

  void ExpectDefaultChangeFailure(const OpDef& old_op_def,
                                  const string& compatibility_error) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("compatibility_error: \"" + compatibility_error + "\"");
   MHTracer_DTPStensorflowPScorePSframeworkPSop_compatibility_testDTcc mht_9(mht_9_v, 371, "", "./tensorflow/core/framework/op_compatibility_test.cc", "ExpectDefaultChangeFailure");

    // This should be all that is needed to get compatibility.
    const OpDef* new_op_def = RegisteredOpDef();
    AddDefaultsToNodeDef(*new_op_def, node_def());

    // Validate that the NodeDef is valid.
    TF_ASSERT_OK(ValidateNodeDef(*node_def(), *new_op_def));

    Status status = OpDefAttrDefaultsUnchanged(old_op_def, *new_op_def);
    if (status.ok()) {
      ADD_FAILURE() << SummarizeOpDef(old_op_def) << " vs. "
                    << SummarizeOpDef(*new_op_def);
    } else {
      EXPECT_TRUE(
          absl::StrContains(status.error_message(), compatibility_error))
          << status << " does not contain " << compatibility_error;
    }
  }
};

// Should be compatible if the Op hasn't changed (sanity check).
REGISTER_OP("Same")
    .Input("a: int32")
    .Input("b: T")
    .Input("c: N * int32")
    .Input("d: N * T")
    .Input("e: TList")
    .Output("ndef: string")
    .Attr("T: type")
    .Attr("N: int")
    .Attr("TList: list(type)");
REGISTER_KERNEL_BUILDER(Name("Same").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, Same) {
  TF_ASSERT_OK(NodeDefBuilder("same", "Same")
                   .Input(FakeInput())
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(3))
                   .Input(FakeInput(3, DT_FLOAT))
                   .Input(FakeInput(2, DT_BOOL))
                   .Finalize(node_def()));
  ExpectSuccess(*RegisteredOpDef());
  EXPECT_EQ(
      "{{node same}} = Same[N=3, T=DT_FLOAT, TList=[DT_BOOL, DT_BOOL]](a, b, "
      "c, c:1, c:2, d, d:1, d:2, e, e:1)",
      Result());
}

// Should be able to add an attr with a default.
REGISTER_OP("AddAttr").Output("ndef: string").Attr("a: int = 42");
REGISTER_KERNEL_BUILDER(Name("AddAttr").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddAttr) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(
      OpDefBuilder("AddAttr").Output("ndef: string").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("add_attr", &old_op.op_def).Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node add_attr}} = AddAttr[a=42]()", Result());
}

// Should be able to make an attr restriction less strict.
REGISTER_OP("LessStrict").Output("ndef: string").Attr("a: {'A', 'B', 'C'}");
REGISTER_KERNEL_BUILDER(Name("LessStrict").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, LessStrict) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("LessStrict")
                   .Output("ndef: string")
                   .Attr("a: {'A', 'B'}")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("less_strict", &old_op.op_def)
                   .Attr("a", "B")
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node less_strict}} = LessStrict[a=\"B\"]()", Result());
}

// Should be able to remove an attr restriction.
REGISTER_OP("RemoveRestriction").Output("ndef: string").Attr("a: type");
REGISTER_KERNEL_BUILDER(Name("RemoveRestriction").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, RemoveRestriction) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("RemoveRestriction")
                   .Output("ndef: string")
                   .Attr("a: {int32, bool}")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("remove_restriction", &old_op.op_def)
                   .Attr("a", DT_INT32)
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node remove_restriction}} = RemoveRestriction[a=DT_INT32]()",
            Result());
}

// Should be able to change the order of attrs.
REGISTER_OP("AttrOrder").Output("ndef: string").Attr("a: int").Attr("b: bool");
REGISTER_KERNEL_BUILDER(Name("AttrOrder").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AttrOrder) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AttrOrder")
                   .Output("ndef: string")
                   .Attr("b: bool")
                   .Attr("a: int")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("attr_order", &old_op.op_def)
                   .Attr("b", true)
                   .Attr("a", 7)
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node attr_order}} = AttrOrder[a=7, b=true]()", Result());
}

// Should be able to make an input/output polymorphic.
// Changing from int32 -> T (where T: type = DT_INT32 by default).
REGISTER_OP("TypePolymorphic")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: type = DT_INT32");
REGISTER_KERNEL_BUILDER(Name("TypePolymorphic").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, TypePolymorphic) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("TypePolymorphic")
                   .Input("a: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("type_polymorphic", &old_op.op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node type_polymorphic}} = TypePolymorphic[T=DT_INT32](a)",
            Result());
}

// Should be able to make a single input/output into a list.
// Changing from int32 -> N * int32 (where N: int = 1 by default).
REGISTER_OP("MakeList")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int = 1");
REGISTER_KERNEL_BUILDER(Name("MakeList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, MakeList) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("MakeList")
                   .Input("a: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("make_list", &old_op.op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node make_list}} = MakeList[N=1](a)", Result());
}

// Should be able to make a single input/output into a polymorphic list.
// Changing from int32 -> N * T (where N: int = 1 by default and
//                                     T: type = DT_INT32 by default).
REGISTER_OP("MakePolyList")
    .Input("a: N * T")
    .Output("ndef: string")
    .Attr("N: int = 1")
    .Attr("T: type = DT_INT32");
REGISTER_KERNEL_BUILDER(Name("MakePolyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, MakePolyList) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("MakePolyList")
                   .Input("a: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("make_poly_list", &old_op.op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node make_poly_list}} = MakePolyList[N=1, T=DT_INT32](a)",
            Result());
}

// Should be able to make a single input/output into an arbitrary list.
// Changing from int32 -> T (where T: list(type) = [DT_INT32] by default).
REGISTER_OP("MakeAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) = [DT_INT32]");
REGISTER_KERNEL_BUILDER(Name("MakeAnyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, MakeAnyList) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("MakeAnyList")
                   .Input("a: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("make_any_list", &old_op.op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node make_any_list}} = MakeAnyList[T=[DT_INT32]](a)", Result());
}

// Should be able to make a single polymorphic input/output into a list of
// the same type.  Changing from T -> N * T (where N: int = 1 by default).
REGISTER_OP("PolyIntoList")
    .Input("a: N * T")
    .Output("ndef: string")
    .Attr("N: int = 1")
    .Attr("T: type");
REGISTER_KERNEL_BUILDER(Name("PolyIntoList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, PolyIntoList) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("PolyIntoList")
                   .Input("a: T")
                   .Output("ndef: string")
                   .Attr("T: type")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("poly_into_list", &old_op.op_def)
                   .Input(FakeInput(DT_INT32))
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node poly_into_list}} = PolyIntoList[N=1, T=DT_INT32](a)",
            Result());
}

// Should be able to make a multiple inputs/outputs into a list with
// the default types matching the inputs/outputs being replaced.

// Changing from int32, int32 -> N * int32 (where N: int = 2 by default).
REGISTER_OP("MakeMultipleSameList")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int = 2");
REGISTER_KERNEL_BUILDER(Name("MakeMultipleSameList").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, MakeMultipleSameList) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("MakeMultipleSameList")
                   .Input("a: int32")
                   .Input("b: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("make_list", &old_op.op_def)
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node make_list}} = MakeMultipleSameList[N=2](a, b)", Result());
}

// Changing from int32, float -> T
// (where T: list(type) = [int32, float] by default).
REGISTER_OP("MakeMultipleAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) = [DT_INT32, DT_FLOAT]");
REGISTER_KERNEL_BUILDER(Name("MakeMultipleAnyList").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, MakeMultipleAnyList) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("MakeMultipleAnyList")
                   .Input("a: int32")
                   .Input("b: float")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("make_list", &old_op.op_def)
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ(
      "{{node make_list}} = MakeMultipleAnyList[T=[DT_INT32, DT_FLOAT]](a, b)",
      Result());
}

// Should be able to change the name of an input/output.
REGISTER_OP("ChangeName").Input("y: int32").Output("ndef: string");
REGISTER_KERNEL_BUILDER(Name("ChangeName").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, ChangeName) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("ChangeName")
                   .Input("x: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("change_name", &old_op.op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node change_name}} = ChangeName[](a)", Result());
}

// Should be able to add an input/output of type
// N * int32 (where N: int = 0 by default).
REGISTER_OP("AddNInts")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int >= 0 = 0");
REGISTER_KERNEL_BUILDER(Name("AddNInts").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddNInts) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(
      OpDefBuilder("AddNInts").Output("ndef: string").Finalize(&old_op));
  TF_ASSERT_OK(
      NodeDefBuilder("add_n_ints", &old_op.op_def).Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node add_n_ints}} = AddNInts[N=0]()", Result());
}

// Should be able to add an input/output of type N * T
// (where N: int = 0 by default, and T: type = any valid default).
REGISTER_OP("AddNSame")
    .Input("a: N * T")
    .Output("ndef: string")
    .Attr("N: int >= 0 = 0")
    .Attr("T: type = DT_BOOL");
REGISTER_KERNEL_BUILDER(Name("AddNSame").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddNSame) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(
      OpDefBuilder("AddNSame").Output("ndef: string").Finalize(&old_op));
  TF_ASSERT_OK(
      NodeDefBuilder("add_n_same", &old_op.op_def).Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node add_n_same}} = AddNSame[N=0, T=DT_BOOL]()", Result());
}

// Should be able to add an input/output of type N * T
// (where N: int >= 0 = 0 by default, and T an existing type attr).
REGISTER_OP("AddNSameAsExisting")
    .Input("a: T")
    .Input("b: N * T")
    .Output("ndef: string")
    .Attr("N: int >= 0 = 0")
    .Attr("T: type");
REGISTER_KERNEL_BUILDER(Name("AddNSameAsExisting").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, AddNSameAsExisting) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AddNSameAsExisting")
                   .Input("a: T")
                   .Output("ndef: string")
                   .Attr("T: type")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("add_n_same_as_existing", &old_op.op_def)
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ(
      "{{node add_n_same_as_existing}} = AddNSameAsExisting[N=0, "
      "T=DT_STRING](a)",
      Result());
}

// Should be able to add an input/output of type T
// (where T: list(type) >= 0 = [] by default).
REGISTER_OP("AddAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) >= 0 = []");
REGISTER_KERNEL_BUILDER(Name("AddAnyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddAnyList) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(
      OpDefBuilder("AddAnyList").Output("ndef: string").Finalize(&old_op));
  TF_ASSERT_OK(
      NodeDefBuilder("add_any_list", &old_op.op_def).Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node add_any_list}} = AddAnyList[T=[]]()", Result());
}

// Should be able to allow shorter lists.
REGISTER_OP("ShorterAnyList")
    .Input("a: T")
    .Output("ndef: string")
    .Attr("T: list(type) >= 1");
REGISTER_KERNEL_BUILDER(Name("ShorterAnyList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, ShorterAnyList) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("ShorterAnyList")
                   .Input("a: T")
                   .Output("ndef: string")
                   .Attr("T: list(type) >= 2")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("shorter_any_list", &old_op.op_def)
                   .Input(FakeInput(2, DT_BOOL))
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ(
      "{{node shorter_any_list}} = ShorterAnyList[T=[DT_BOOL, DT_BOOL]](a, "
      "a:1)",
      Result());
}

REGISTER_OP("ShorterSameList")
    .Input("a: N * int32")
    .Output("ndef: string")
    .Attr("N: int >= 1");
REGISTER_KERNEL_BUILDER(Name("ShorterSameList").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, ShorterSameList) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("ShorterSameList")
                   .Input("a: N * int32")
                   .Output("ndef: string")
                   .Attr("N: int >= 2")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("shorter_same_list", &old_op.op_def)
                   .Input(FakeInput(2))
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node shorter_same_list}} = ShorterSameList[N=2](a, a:1)",
            Result());
}

// Can remove a restriction to an attr

REGISTER_OP("AttrRemoveRestriction").Attr("t: type").Output("ndef: string");
REGISTER_KERNEL_BUILDER(Name("AttrRemoveRestriction").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, AttrRemoveRestriction) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AttrRemoveRestriction")
                   .Attr("t: {int32,int64}")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("remove_restriction", &old_op.op_def)
                   .Attr("t", DT_INT32)
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node remove_restriction}} = AttrRemoveRestriction[t=DT_INT32]()",
            Result());
}

// Can make the restrictions on an attr less restrictive.

REGISTER_OP("AttrLessRestrictive")
    .Attr("t: {int32, int64, bool}")
    .Output("ndef: string");
REGISTER_KERNEL_BUILDER(Name("AttrLessRestrictive").Device(DEVICE_CPU),
                        TestKernel);

TEST_F(OpCompatibilityTest, AttrLessRestrictive) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AttrLessRestrictive")
                   .Attr("t: {int32, int64}")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("less_restrictive", &old_op.op_def)
                   .Attr("t", DT_INT32)
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node less_restrictive}} = AttrLessRestrictive[t=DT_INT32]()",
            Result());
}

// Can remove a minimum from an attr.

REGISTER_OP("AttrRemoveMin").Attr("n: int").Output("ndef: string");
REGISTER_KERNEL_BUILDER(Name("AttrRemoveMin").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AttrRemoveMin) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AttrRemoveMin")
                   .Attr("n: int >= 3")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("remove_min", &old_op.op_def)
                   .Attr("n", 4)
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node remove_min}} = AttrRemoveMin[n=4]()", Result());
}

// Can lower the minimum on an attr.

REGISTER_OP("AttrLowerMin").Attr("n: int >= 1").Output("ndef: string");
REGISTER_KERNEL_BUILDER(Name("AttrLowerMin").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AttrLowerMin) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AttrLowerMin")
                   .Attr("n: int >= 3")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("lower_min", &old_op.op_def)
                   .Attr("n", 4)
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node lower_min}} = AttrLowerMin[n=4]()", Result());
}

// Can make a ref input into a non-ref input.

REGISTER_OP("InputRemoveRef").Input("i: int32").Output("ndef: string");
REGISTER_KERNEL_BUILDER(Name("InputRemoveRef").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, InputRemoveRef) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("InputRemoveRef")
                   .Input("i: Ref(int32)")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("remove_input_ref", &old_op.op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
}

// Can make a non-ref output into a ref output.

REGISTER_OP("OutputAddRef").Output("o: Ref(int32)").Output("ndef: string");
REGISTER_KERNEL_BUILDER(Name("OutputAddRef").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, OutputAddRef) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("OutputAddRef")
                   .Output("o: int32")
                   .Output("ndef: string")
                   .Finalize(&old_op));
  TF_ASSERT_OK(
      NodeDefBuilder("add_output_ref", &old_op.op_def).Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
}

// Negative tests -------------------------------------------------------------

// Can't add an attr without a default.
REGISTER_OP("AddAttrNoDefault").Attr("a: int");

TEST_F(OpCompatibilityTest, AddAttrNoDefaultFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AddAttrNoDefault").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def).Finalize(node_def()));
  ExpectInvalid(old_op.op_def, "NodeDef missing attr 'a'",
                "Attr 'a' added without default");
}

// Can't add a non-list input/output.
REGISTER_OP("AddSingleInput").Input("a: int32");

TEST_F(OpCompatibilityTest, AddSingleInputFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AddSingleInput").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def).Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "expected inputs 'int32' do not match 0 inputs specified",
                "Input signature mismatch '' vs. 'int32'");
}

// Can't add a list input/output without an empty default.

REGISTER_OP("AddNIntsBigDefault").Input("a: N * int32").Attr("N: int = 1");
REGISTER_OP("AddNSameBigDefault")
    .Input("a: N * T")
    .Attr("N: int = 1")
    .Attr("T: type = DT_INT32");
REGISTER_OP("AddListBigDefault")
    .Input("a: T")
    .Attr("T: list(type) = [DT_INT32]");

TEST_F(OpCompatibilityTest, AddNIntsBigDefaultFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AddNIntsBigDefault").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def).Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "expected inputs 'int32' do not match 0 inputs specified",
                "Input signature mismatch '' vs. 'int32'");
}

TEST_F(OpCompatibilityTest, AddNSameBigDefaultFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AddNSameBigDefault").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def).Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "expected inputs 'int32' do not match 0 inputs specified",
                "Input signature mismatch '' vs. 'int32'");
}

TEST_F(OpCompatibilityTest, AddListBigDefaultFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AddListBigDefault").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def).Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "expected inputs 'int32' do not match 0 inputs specified",
                "Input signature mismatch '' vs. 'int32'");
}

// Can't change the type of an input/output.

REGISTER_OP("ChangeType").Input("a: float");

TEST_F(OpCompatibilityTest, ChangeTypeFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("ChangeType").Input("a: int32").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectTypeMismatch(old_op.op_def,
                     "Input signature mismatch 'int32' vs. 'float'");
}

// Can't change the order of inputs/outputs.

REGISTER_OP("ChangeOrder").Input("a: float").Input("b: int32");

TEST_F(OpCompatibilityTest, ChangeOrderFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("ChangeOrder")
                   .Input("b: int32")
                   .Input("a: float")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def)
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectTypeMismatch(
      old_op.op_def,
      "Input signature mismatch 'int32, float' vs. 'float, int32'");
}

// Can't remove inputs/outputs.

REGISTER_OP("RemoveInput");

TEST_F(OpCompatibilityTest, RemoveInputFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("RemoveInput").Input("a: float").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "expected inputs '' do not match 1 inputs specified",
                "Input signature mismatch 'float' vs. ''");
}

// Can't change the type of an attr.

REGISTER_OP("ChangeAttrType").Attr("a: int");

TEST_F(OpCompatibilityTest, ChangeAttrTypeFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(
      OpDefBuilder("ChangeAttrType").Attr("a: bool").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def)
                   .Attr("a", true)
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def, "value with type 'bool' when 'int' expected",
                "Attr 'a' changed type 'bool' -> 'int'");
}

// Can't change an attr from a list.

REGISTER_OP("AttrFromList").Attr("a: int");

TEST_F(OpCompatibilityTest, AttrFromListFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(
      OpDefBuilder("AttrFromList").Attr("a: list(int)").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def)
                   .Attr("a", {5})
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "value with type 'list(int)' when 'int' expected",
                "Attr 'a' changed type 'list(int)' -> 'int'");
}

// Can't change an attr to a list.

REGISTER_OP("AttrToList").Attr("a: list(int)");

TEST_F(OpCompatibilityTest, AttrToListFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AttrToList").Attr("a: int").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def)
                   .Attr("a", 5)
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "value with type 'int' when 'list(int)' expected",
                "Attr 'a' changed type 'int' -> 'list(int)'");
}

// Can't change an input from polymorphic to a list of any type.

REGISTER_OP("PolymorphicToAnyList").Input("a: T").Attr("T: list(type)");

TEST_F(OpCompatibilityTest, PolymorphicToAnyListFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("PolymorphicToAnyList")
                   .Input("a: T")
                   .Attr("T: type")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def)
                   .Input(FakeInput(DT_INT32))
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "value with type 'type' when 'list(type)' expected",
                "Attr 'T' changed type 'type' -> 'list(type)'");
}

// Can't change an input from a list of the same type to a list of any type.

REGISTER_OP("SameToAnyList")
    .Input("a: T")
    .Attr("T: list(type)")
    .Attr("N: int = 1");

TEST_F(OpCompatibilityTest, SameToAnyListFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("SameToAnyList")
                   .Input("a: N * T")
                   .Attr("T: type")
                   .Attr("N: int")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("fails", &old_op.op_def)
                   .Input(FakeInput(1, DT_INT32))
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "value with type 'type' when 'list(type)' expected",
                "Attr 'T' changed type 'type' -> 'list(type)'");
}

// Can't add a restriction to an attr

REGISTER_OP("AttrAddRestriction").Attr("t: {int32, int64}");

TEST_F(OpCompatibilityTest, AttrAddRestrictionFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(
      OpDefBuilder("AttrAddRestriction").Attr("t: type").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("add_restriction", &old_op.op_def)
                   .Attr("t", DT_BOOL)
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "Value for attr 't' of bool is not in the list of allowed "
                "values: int32, int64",
                "Attr 't' has a stricter set of allowed values; from "
                "no restriction to [DT_INT32, DT_INT64]");
}

// Can't make the restrictions on an attr more restrictive.

REGISTER_OP("AttrMoreRestrictive").Attr("t: {int32, int64}");

TEST_F(OpCompatibilityTest, AttrMoreRestrictiveFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AttrMoreRestrictive")
                   .Attr("t: {int32, int64, bool}")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("more_restrictive", &old_op.op_def)
                   .Attr("t", DT_BOOL)
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "Value for attr 't' of bool is not in the list of allowed "
                "values: int32, int64",
                "Attr 't' has a stricter set of allowed values; from "
                "[DT_INT32, DT_INT64, DT_BOOL] to [DT_INT32, DT_INT64]");
}

// Can't add a minimum to an attr.

REGISTER_OP("AttrAddMin").Attr("n: int >= 3");

TEST_F(OpCompatibilityTest, AttrAddMinFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AttrAddMin").Attr("n: int").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("add_min", &old_op.op_def)
                   .Attr("n", 2)
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "Value for attr 'n' of 2 must be at least minimum 3",
                "Attr 'n' has a higher minimum; from no minimum to 3");
}

// Can't raise the minimum on an attr.

REGISTER_OP("AttrRaiseMin").Attr("n: int >= 3");

TEST_F(OpCompatibilityTest, AttrRaiseMinFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(
      OpDefBuilder("AttrRaiseMin").Attr("n: int >= 1").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("raise_min", &old_op.op_def)
                   .Attr("n", 2)
                   .Finalize(node_def()));
  ExpectInvalid(old_op.op_def,
                "Value for attr 'n' of 2 must be at least minimum 3",
                "Attr 'n' has a higher minimum; from 1 to 3");
}

// Can't make a non-ref input into a ref input.

REGISTER_OP("InputAddRef").Input("i: Ref(int32)");

TEST_F(OpCompatibilityTest, InputAddRefFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("InputAddRef").Input("i: int32").Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("add_input_ref", &old_op.op_def)
                   .Input(FakeInput())
                   .Finalize(node_def()));
  ExpectTypeMismatch(old_op.op_def, "Input 0 changed from non-ref to ref");
}

// Can't make a ref output into a non-ref output.

REGISTER_OP("OutputRemoveRef").Output("o: int32");

TEST_F(OpCompatibilityTest, OutputRemoveRefFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("OutputRemoveRef")
                   .Output("o: Ref(int32)")
                   .Finalize(&old_op));
  TF_ASSERT_OK(
      NodeDefBuilder("remove_output_ref", &old_op.op_def).Finalize(node_def()));
  ExpectTypeMismatch(old_op.op_def, "Output 0 changed from ref to non-ref");
}

// Can't rename an output, to avoid problems in FunctionDefs.

REGISTER_OP("RenameOutput").Output("new: int32");

TEST_F(OpCompatibilityTest, RenameOutputFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(
      OpDefBuilder("RenameOutput").Output("old: int32").Finalize(&old_op));
  TF_ASSERT_OK(
      NodeDefBuilder("rename_output", &old_op.op_def).Finalize(node_def()));
  ExpectRenameFailure(old_op.op_def,
                      "Output signature mismatch 'old:int32' vs. 'new:int32'");
}

REGISTER_OP("RenameNOutputs").Output("new: N*int32").Attr("N: int");

TEST_F(OpCompatibilityTest, RenameNOutputsFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("RenameNOutputs")
                   .Output("old: N*int32")
                   .Attr("N: int")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("rename_n_outputs", &old_op.op_def)
                   .Attr("N", 2)
                   .Finalize(node_def()));
  ExpectRenameFailure(
      old_op.op_def,
      "Output signature mismatch 'old:N * int32' vs. 'new:N * int32'");
}

REGISTER_OP("RenameOutputList").Output("new: T").Attr("T: list(type)");

TEST_F(OpCompatibilityTest, RenameOutputListFails) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("RenameOutputList")
                   .Output("old: T")
                   .Attr("T: list(type)")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("rename_output_list", &old_op.op_def)
                   .Attr("T", {DT_INT32, DT_FLOAT})
                   .Finalize(node_def()));
  ExpectRenameFailure(old_op.op_def,
                      "Output signature mismatch 'old:T' vs. 'new:T'");
}

// It's ok to add a default to an attr if it doesn't already have one.
REGISTER_OP("AddDefault").Output("ndef: string").Attr("a: int = 1234");
REGISTER_KERNEL_BUILDER(Name("AddDefault").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, AddDefault) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("AddDefault")
                   .Output("ndef: string")
                   .Attr("a: int")
                   .Finalize(&old_op));
  TF_ASSERT_OK(NodeDefBuilder("add_default", &old_op.op_def)
                   .Attr("a", 765)
                   .Finalize(node_def()));
  ExpectSuccess(old_op.op_def);
  EXPECT_EQ("{{node add_default}} = AddDefault[a=765]()", Result());
}

// Should not be able to remove a default from an attr.
REGISTER_OP("RemoveDefault").Output("ndef: string").Attr("a: int");
REGISTER_KERNEL_BUILDER(Name("RemoveDefault").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, RemoveDefault) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("RemoveDefault")
                   .Output("ndef: string")
                   .Attr("a: int = 91")
                   .Finalize(&old_op));
  TF_ASSERT_OK(
      NodeDefBuilder("remove_default", &old_op.op_def).Finalize(node_def()));
  ExpectDefaultChangeFailure(
      old_op.op_def,
      "Attr 'a' has removed it's default; from 91 to no default");
}

// Should not be able to change a default for an attr.
REGISTER_OP("ChangeDefault").Output("ndef: string").Attr("a: int = 1");
REGISTER_KERNEL_BUILDER(Name("ChangeDefault").Device(DEVICE_CPU), TestKernel);

TEST_F(OpCompatibilityTest, ChangeDefault) {
  OpRegistrationData old_op;
  TF_ASSERT_OK(OpDefBuilder("ChangeDefault")
                   .Output("ndef: string")
                   .Attr("a: int = 2")
                   .Finalize(&old_op));
  TF_ASSERT_OK(
      NodeDefBuilder("change_default", &old_op.op_def).Finalize(node_def()));
  ExpectDefaultChangeFailure(
      old_op.op_def, "Attr 'a' has changed it's default value; from 2 to 1");
}

}  // namespace
}  // namespace tensorflow
