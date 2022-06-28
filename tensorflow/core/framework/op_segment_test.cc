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
class MHTracer_DTPStensorflowPScorePSframeworkPSop_segment_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segment_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSop_segment_testDTcc() {
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

#include "tensorflow/core/framework/op_segment.h"

#include <vector>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class OpSegmentTest : public ::testing::Test {
 protected:
  DeviceBase device_;
  std::vector<NodeDef> int32_nodedefs_;
  std::vector<NodeDef> float_nodedefs_;

  OpSegmentTest() : device_(Env::Default()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segment_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/core/framework/op_segment_test.cc", "OpSegmentTest");

    for (int i = 0; i < 10; ++i) {
      NodeDef def;
      TF_CHECK_OK(NodeDefBuilder(strings::StrCat("op", i), "Mul")
                      .Input("x", 0, DT_INT32)
                      .Input("y", 0, DT_INT32)
                      .Finalize(&def));
      int32_nodedefs_.push_back(def);
      TF_CHECK_OK(NodeDefBuilder(strings::StrCat("op", i), "Mul")
                      .Input("x", 0, DT_FLOAT)
                      .Input("y", 0, DT_FLOAT)
                      .Finalize(&def));
      float_nodedefs_.push_back(def);
    }
  }

  void ValidateOpAndTypes(OpKernel* op, const NodeDef& expected, DataType dt) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segment_testDTcc mht_1(mht_1_v, 226, "", "./tensorflow/core/framework/op_segment_test.cc", "ValidateOpAndTypes");

    ASSERT_NE(op, nullptr);
    EXPECT_EQ(expected.DebugString(), op->def().DebugString());
    EXPECT_EQ(2, op->num_inputs());
    EXPECT_EQ(dt, op->input_type(0));
    EXPECT_EQ(dt, op->input_type(1));
    EXPECT_EQ(1, op->num_outputs());
    EXPECT_EQ(dt, op->output_type(0));
  }

  OpSegment::CreateKernelFn GetFn(const NodeDef* ndef) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segment_testDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/framework/op_segment_test.cc", "GetFn");

    return [this, ndef](OpKernel** kernel) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segment_testDTcc mht_3(mht_3_v, 243, "", "./tensorflow/core/framework/op_segment_test.cc", "lambda");

      Status s;
      auto created = CreateOpKernel(DEVICE_CPU, &device_, cpu_allocator(),
                                    *ndef, TF_GRAPH_DEF_VERSION, &s);
      if (s.ok()) {
        *kernel = created.release();
      }
      return s;
    };
  }
};

TEST_F(OpSegmentTest, Basic) {
  OpSegment opseg;
  OpKernel* op;

  opseg.AddHold("A");
  opseg.AddHold("B");
  for (int i = 0; i < 10; ++i) {
    // Register in session A.
    auto* ndef = &float_nodedefs_[i];
    TF_EXPECT_OK(opseg.FindOrCreate("A", ndef->name(), &op, GetFn(ndef)));
    ValidateOpAndTypes(op, *ndef, DT_FLOAT);

    // Register in session B.
    ndef = &int32_nodedefs_[i];
    TF_EXPECT_OK(opseg.FindOrCreate("B", ndef->name(), &op, GetFn(ndef)));
    ValidateOpAndTypes(op, *ndef, DT_INT32);
  }

  auto reterr = [](OpKernel** kernel) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSop_segment_testDTcc mht_4(mht_4_v, 276, "", "./tensorflow/core/framework/op_segment_test.cc", "lambda");

    return errors::Internal("Should not be called");
  };
  for (int i = 0; i < 10; ++i) {
    // Lookup op in session A.
    TF_EXPECT_OK(
        opseg.FindOrCreate("A", strings::StrCat("op", i), &op, reterr));
    ValidateOpAndTypes(op, float_nodedefs_[i], DT_FLOAT);

    // Lookup op in session B.
    TF_EXPECT_OK(
        opseg.FindOrCreate("B", strings::StrCat("op", i), &op, reterr));
    ValidateOpAndTypes(op, int32_nodedefs_[i], DT_INT32);
  }

  opseg.RemoveHold("A");
  opseg.RemoveHold("B");
}

TEST_F(OpSegmentTest, SessionNotFound) {
  OpSegment opseg;
  OpKernel* op;
  NodeDef def = float_nodedefs_[0];
  Status s = opseg.FindOrCreate("A", def.name(), &op, GetFn(&def));
  EXPECT_TRUE(errors::IsNotFound(s)) << s;
}

TEST_F(OpSegmentTest, CreateFailure) {
  OpSegment opseg;
  OpKernel* op;
  NodeDef def = float_nodedefs_[0];
  def.set_op("nonexistop");
  opseg.AddHold("A");
  Status s = opseg.FindOrCreate("A", def.name(), &op, GetFn(&def));
  EXPECT_TRUE(errors::IsNotFound(s)) << s;
  opseg.RemoveHold("A");
}

TEST_F(OpSegmentTest, AddRemoveHolds) {
  OpSegment opseg;
  OpKernel* op;
  const auto& ndef = int32_nodedefs_[0];

  // No op.
  opseg.RemoveHold("null");

  // Thread1 register the op and wants to ensure it alive.
  opseg.AddHold("foo");
  TF_EXPECT_OK(opseg.FindOrCreate("foo", ndef.name(), &op, GetFn(&ndef)));

  // Thread2 starts some execution needs "op" to be alive.
  opseg.AddHold("foo");

  // Thread1 clears session "foo".  E.g., a master sends CleanupGraph
  // before an execution finishes.
  opseg.RemoveHold("foo");

  // Thread2 should still be able to access "op".
  ValidateOpAndTypes(op, ndef, DT_INT32);

  // Thread2 then remove its hold on "foo".
  opseg.RemoveHold("foo");
}

}  // namespace tensorflow
