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
class MHTracer_DTPStensorflowPScorePSframeworkPSmemory_types_testDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_types_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPSmemory_types_testDTcc() {
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

#include "tensorflow/core/framework/memory_types.h"

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class DummyKernel : public OpKernel {
 public:
  explicit DummyKernel(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_types_testDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/framework/memory_types_test.cc", "DummyKernel");
}
  void Compute(tensorflow::OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPSmemory_types_testDTcc mht_1(mht_1_v, 203, "", "./tensorflow/core/framework/memory_types_test.cc", "Compute");
}
};

REGISTER_OP("HostMemoryTest")
    .Input("a: float")
    .Input("b: float")
    .Input("c: T")
    .Input("d: N * string")
    .Input("e: Tlist")
    .Input("f: Rlist")
    .Output("o: N * T")
    .Output("p: N * T")
    .Output("r: Tlist")
    .Attr("T: type")
    .Attr("N: int")
    .Attr("Tlist: list(type)")
    .Attr("Rlist: list(type)");
REGISTER_KERNEL_BUILDER(Name("HostMemoryTest").Device(DEVICE_CPU), DummyKernel);
REGISTER_KERNEL_BUILDER(Name("HostMemoryTest")
                            .Device(DEVICE_GPU)
                            .HostMemory("b")
                            .HostMemory("d")
                            .HostMemory("e")
                            .HostMemory("p"),
                        DummyKernel);

TEST(MemoryTypesForNode, Simple) {
  NodeDef node_def;
  TF_ASSERT_OK(NodeDefBuilder("test", "HostMemoryTest")
                   .Input(FakeInput())
                   .Input(FakeInput())
                   .Input(FakeInput(DT_BOOL))
                   .Input(FakeInput(3))
                   .Input(FakeInput({DT_INT32, DT_FLOAT, DT_INT32}))
                   .Input(FakeInput({DT_RESOURCE, DT_STRING, DT_RESOURCE}))
                   .Finalize(&node_def));
  AddNodeAttr("_input_hostmem", {0}, &node_def);
  AddNodeAttr("_output_hostmem", {6, 7}, &node_def);

  MemoryTypeVector input, output;

  TF_EXPECT_OK(MemoryTypesForNode(OpRegistry::Global(), DEVICE_CPU, node_def,
                                  &input, &output));
  // a:float, b:bool, c:3*string, d:(int32, float, int32),
  // e:(resource, string, resource)
  EXPECT_EQ(
      MemoryTypeVector({HOST_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY, HOST_MEMORY,
                        HOST_MEMORY, HOST_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY,
                        DEVICE_MEMORY, HOST_MEMORY, HOST_MEMORY, HOST_MEMORY}),
      input);
  // o:3*bool, p:(int32, float, int32)
  EXPECT_EQ(MemoryTypeVector({DEVICE_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY,
                              DEVICE_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY,
                              HOST_MEMORY, HOST_MEMORY, DEVICE_MEMORY}),
            output);

  TF_EXPECT_OK(MemoryTypesForNode(OpRegistry::Global(), DEVICE_GPU, node_def,
                                  &input, &output));
  EXPECT_EQ(
      MemoryTypeVector({HOST_MEMORY, HOST_MEMORY, DEVICE_MEMORY, HOST_MEMORY,
                        HOST_MEMORY, HOST_MEMORY, HOST_MEMORY, HOST_MEMORY,
                        HOST_MEMORY, HOST_MEMORY, HOST_MEMORY, HOST_MEMORY}),
      input);
  EXPECT_EQ(MemoryTypeVector({DEVICE_MEMORY, DEVICE_MEMORY, DEVICE_MEMORY,
                              HOST_MEMORY, HOST_MEMORY, HOST_MEMORY,
                              HOST_MEMORY, HOST_MEMORY, DEVICE_MEMORY}),
            output);
}

}  // namespace tensorflow
