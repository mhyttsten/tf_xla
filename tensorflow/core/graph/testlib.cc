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
class MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc() {
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

#include "tensorflow/core/graph/testlib.h"

#include <vector>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace test {
namespace graph {

Node* Send(Graph* g, Node* input, const string& tensor, const string& sender,
           const uint64 sender_incarnation, const string& receiver) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("tensor: \"" + tensor + "\"");
   mht_0_v.push_back("sender: \"" + sender + "\"");
   mht_0_v.push_back("receiver: \"" + receiver + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_0(mht_0_v, 208, "", "./tensorflow/core/graph/testlib.cc", "Send");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_Send")
                  .Input(input, 0)
                  .Attr("tensor_name", tensor)
                  .Attr("send_device", sender)
                  .Attr("send_device_incarnation",
                        static_cast<int64_t>(sender_incarnation))
                  .Attr("recv_device", receiver)
                  .Finalize(g, &ret));
  return ret;
}

Node* Recv(Graph* g, const string& tensor, const string& type,
           const string& sender, const uint64 sender_incarnation,
           const string& receiver) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("tensor: \"" + tensor + "\"");
   mht_1_v.push_back("type: \"" + type + "\"");
   mht_1_v.push_back("sender: \"" + sender + "\"");
   mht_1_v.push_back("receiver: \"" + receiver + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_1(mht_1_v, 230, "", "./tensorflow/core/graph/testlib.cc", "Recv");

  Node* ret;
  DataType dtype;
  CHECK(DataTypeFromString(type, &dtype));
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_Recv")
                  .Attr("tensor_type", dtype)
                  .Attr("tensor_name", tensor)
                  .Attr("send_device", sender)
                  .Attr("send_device_incarnation",
                        static_cast<int64_t>(sender_incarnation))
                  .Attr("recv_device", receiver)
                  .Finalize(g, &ret));
  return ret;
}

Node* Constant(Graph* g, const Tensor& tensor) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_2(mht_2_v, 248, "", "./tensorflow/core/graph/testlib.cc", "Constant");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Const")
                  .Attr("dtype", tensor.dtype())
                  .Attr("value", tensor)
                  .Finalize(g, &ret));
  return ret;
}

Node* Constant(Graph* g, const Tensor& tensor, const string& name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_3(mht_3_v, 261, "", "./tensorflow/core/graph/testlib.cc", "Constant");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(name, "Const")
                  .Attr("dtype", tensor.dtype())
                  .Attr("value", tensor)
                  .Finalize(g, &ret));
  return ret;
}

Node* HostConstant(Graph* g, const Tensor& tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_4(mht_4_v, 273, "", "./tensorflow/core/graph/testlib.cc", "HostConstant");

  return HostConstant(g, tensor, g->NewName("n"));
}

Node* HostConstant(Graph* g, const Tensor& tensor, const string& name) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_5(mht_5_v, 281, "", "./tensorflow/core/graph/testlib.cc", "HostConstant");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(name, "HostConst")
                  .Attr("dtype", tensor.dtype())
                  .Attr("value", tensor)
                  .Finalize(g, &ret));
  return ret;
}

Node* Var(Graph* g, const DataType dtype, const TensorShape& shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_6(mht_6_v, 293, "", "./tensorflow/core/graph/testlib.cc", "Var");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Variable")
                  .Attr("dtype", dtype)
                  .Attr("shape", shape)
                  .Finalize(g, &ret));
  return ret;
}

Node* Var(Graph* g, const DataType dtype, const TensorShape& shape,
          const string& name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_7(mht_7_v, 307, "", "./tensorflow/core/graph/testlib.cc", "Var");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(name, "Variable")
                  .Attr("dtype", dtype)
                  .Attr("shape", shape)
                  .Finalize(g, &ret));
  return ret;
}

Node* Assign(Graph* g, Node* var, Node* val) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_8(mht_8_v, 319, "", "./tensorflow/core/graph/testlib.cc", "Assign");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Assign")
                  .Input(var)
                  .Input(val)
                  .Attr("use_locking", true)
                  .Finalize(g, &ret));
  return ret;
}

Node* Cumsum(Graph* g, Node* data, Node* axes, bool exclusive, bool reverse) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_9(mht_9_v, 332, "", "./tensorflow/core/graph/testlib.cc", "Cumsum");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Cumsum")
                  .Input(data)
                  .Input(axes)
                  .Attr("exclusive", exclusive)
                  .Attr("reverse", reverse)
                  .Finalize(g, &ret));
  return ret;
}

Node* Reduce(Graph* g, const string& reduce, Node* data, Node* axes,
             bool keep_dims) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("reduce: \"" + reduce + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_10(mht_10_v, 348, "", "./tensorflow/core/graph/testlib.cc", "Reduce");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), reduce, g->op_registry())
                  .Input(data)
                  .Input(axes)
                  .Attr("keep_dims", keep_dims)
                  .Finalize(g, &ret));
  return ret;
}

Node* QuantizeToUINT8(Graph* g, Node* data) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_11(mht_11_v, 361, "", "./tensorflow/core/graph/testlib.cc", "QuantizeToUINT8");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Quantize")
                  .Input(data)
                  .Attr("T", DT_QUINT8)
                  .Attr("max_range", 1.0f)
                  .Attr("min_range", -1.0f)
                  .Finalize(g, &ret));
  return ret;
}

Node* Matmul(Graph* g, Node* in0, Node* in1, bool transpose_a,
             bool transpose_b) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_12(mht_12_v, 376, "", "./tensorflow/core/graph/testlib.cc", "Matmul");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "MatMul")
                  .Input(in0)
                  .Input(in1)
                  .Attr("transpose_a", transpose_a)
                  .Attr("transpose_b", transpose_b)
                  .Finalize(g, &ret));
  return ret;
}

Node* BatchMatmul(Graph* g, Node* in0, Node* in1, bool adj_x, bool adj_y) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_13(mht_13_v, 390, "", "./tensorflow/core/graph/testlib.cc", "BatchMatmul");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BatchMatMul")
                  .Input(in0)
                  .Input(in1)
                  .Attr("adj_x", adj_x)
                  .Attr("adj_y", adj_y)
                  .Finalize(g, &ret));
  return ret;
}

Node* RandomNumberGenerator(const string& op, Graph* g, Node* input,
                            DataType dtype) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("op: \"" + op + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_14(mht_14_v, 406, "", "./tensorflow/core/graph/testlib.cc", "RandomNumberGenerator");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), op, g->op_registry())
                  .Input(input)
                  .Attr("dtype", dtype)
                  .Attr("seed", 0)
                  .Finalize(g, &ret));
  return ret;
}

Node* RandomUniform(Graph* g, Node* input, DataType dtype) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_15(mht_15_v, 419, "", "./tensorflow/core/graph/testlib.cc", "RandomUniform");

  return RandomNumberGenerator("RandomUniform", g, input, dtype);
}

Node* RandomGaussian(Graph* g, Node* input, DataType dtype) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_16(mht_16_v, 426, "", "./tensorflow/core/graph/testlib.cc", "RandomGaussian");

  return RandomNumberGenerator("RandomStandardNormal", g, input, dtype);
}

Node* TruncatedNormal(Graph* g, Node* input, DataType dtype) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_17(mht_17_v, 433, "", "./tensorflow/core/graph/testlib.cc", "TruncatedNormal");

  return RandomNumberGenerator("TruncatedNormal", g, input, dtype);
}

Node* RandomGamma(Graph* g, Node* shape, Node* alpha) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_18(mht_18_v, 440, "", "./tensorflow/core/graph/testlib.cc", "RandomGamma");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "RandomGamma")
                  .Input(shape)
                  .Input(alpha)
                  .Attr("seed", 0)
                  .Finalize(g, &ret));
  return ret;
}

Node* RandomPoisson(Graph* g, Node* shape, Node* lam) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_19(mht_19_v, 453, "", "./tensorflow/core/graph/testlib.cc", "RandomPoisson");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "RandomPoisson")
                  .Input(shape)
                  .Input(lam)
                  .Attr("seed", 0)
                  .Finalize(g, &ret));
  return ret;
}

Node* Unary(Graph* g, const string& func, Node* input, int index) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_20(mht_20_v, 467, "", "./tensorflow/core/graph/testlib.cc", "Unary");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), func, g->op_registry())
                  .Input(input, index)
                  .Finalize(g, &ret));
  return ret;
}

Node* Binary(Graph* g, const string& func, Node* in0, Node* in1) {
   std::vector<std::string> mht_21_v;
   mht_21_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_21(mht_21_v, 479, "", "./tensorflow/core/graph/testlib.cc", "Binary");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), func, g->op_registry())
                  .Input(in0)
                  .Input(in1)
                  .Finalize(g, &ret));
  return ret;
}

Node* Multi(Graph* g, const string& func, gtl::ArraySlice<Node*> ins) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("func: \"" + func + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_22(mht_22_v, 492, "", "./tensorflow/core/graph/testlib.cc", "Multi");

  Node* ret;
  auto b = NodeBuilder(g->NewName("n"), func, g->op_registry());
  for (Node* n : ins) b = b.Input(n);
  TF_CHECK_OK(b.Finalize(g, &ret));
  return ret;
}

Node* Identity(Graph* g, Node* input, int index) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_23(mht_23_v, 503, "", "./tensorflow/core/graph/testlib.cc", "Identity");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Identity")
                  .Input(input, index)
                  .Finalize(g, &ret));
  return ret;
}

Node* Add(Graph* g, Node* in0, Node* in1) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_24(mht_24_v, 514, "", "./tensorflow/core/graph/testlib.cc", "Add");
 return Binary(g, "Add", in0, in1); }

Node* Reverse(Graph* g, Node* tensor, Node* axis) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_25(mht_25_v, 519, "", "./tensorflow/core/graph/testlib.cc", "Reverse");

  return Binary(g, "ReverseV2", tensor, axis);
}

Node* Roll(Graph* g, Node* input, Node* shift, Node* axis) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_26(mht_26_v, 526, "", "./tensorflow/core/graph/testlib.cc", "Roll");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Roll", g->op_registry())
                  .Input(input)
                  .Input(shift)
                  .Input(axis)
                  .Finalize(g, &ret));
  return ret;
}

Node* Error(Graph* g, Node* input, const string& errmsg, bool log_error) {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("errmsg: \"" + errmsg + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_27(mht_27_v, 540, "", "./tensorflow/core/graph/testlib.cc", "Error");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Error")
                  .Input(input)
                  .Attr("message", errmsg)
                  .Attr("log_error", log_error)
                  .Finalize(g, &ret));
  return ret;
}

Node* InvalidRefType(Graph* g, DataType out_type, DataType invalid_type) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_28(mht_28_v, 553, "", "./tensorflow/core/graph/testlib.cc", "InvalidRefType");

  DCHECK(out_type != invalid_type);
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "InvalidRefType")
                  .Attr("TIn", out_type)
                  .Attr("TOut", invalid_type)
                  .Finalize(g, &ret));
  return ret;
}

Node* Delay(Graph* g, Node* input, Microseconds delay_micros) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_29(mht_29_v, 566, "", "./tensorflow/core/graph/testlib.cc", "Delay");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Delay")
                  .Input(input)
                  .Attr("micros", delay_micros.value())
                  .Finalize(g, &ret));
  return ret;
}

Node* NoOp(Graph* g, const std::vector<Node*>& control_inputs) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_30(mht_30_v, 578, "", "./tensorflow/core/graph/testlib.cc", "NoOp");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "NoOp")
                  .ControlInputs(control_inputs)
                  .Finalize(g, &ret));
  return ret;
}

Node* Switch(Graph* g, Node* in0, Node* in1) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_31(mht_31_v, 589, "", "./tensorflow/core/graph/testlib.cc", "Switch");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Switch")
                  .Input(in0)
                  .Input(in1)
                  .Finalize(g, &ret));
  return ret;
}

Node* Enter(Graph* g, Node* input, const string& frame_name) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("frame_name: \"" + frame_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_32(mht_32_v, 602, "", "./tensorflow/core/graph/testlib.cc", "Enter");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Enter")
                  .Input(input)
                  .Attr("frame_name", frame_name)
                  .Finalize(g, &ret));
  return ret;
}

Node* Exit(Graph* g, Node* input) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_33(mht_33_v, 614, "", "./tensorflow/core/graph/testlib.cc", "Exit");

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("n"), "Exit").Input(input).Finalize(g, &ret));
  return ret;
}

Node* Merge(Graph* g, Node* in0, Node* in1) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_34(mht_34_v, 624, "", "./tensorflow/core/graph/testlib.cc", "Merge");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Merge")
                  .Input({in0, in1})
                  .Finalize(g, &ret));
  return ret;
}

Node* Merge(Graph* g, Node* in0, gtl::ArraySlice<string> remaining_in) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_35(mht_35_v, 635, "", "./tensorflow/core/graph/testlib.cc", "Merge");

  std::vector<NodeBuilder::NodeOut> inputs;
  inputs.reserve(remaining_in.size() + 1);
  inputs.emplace_back(in0);
  for (const string& in_name : remaining_in) {
    inputs.emplace_back(in_name, 0, inputs[0].dt);
  }

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("n"), "Merge").Input(inputs).Finalize(g, &ret));
  return ret;
}

Node* Concat(Graph* g, Node* concat_dim, gtl::ArraySlice<Node*> tensors) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_36(mht_36_v, 652, "", "./tensorflow/core/graph/testlib.cc", "Concat");

  std::vector<NodeBuilder::NodeOut> nodeouts;
  nodeouts.reserve(tensors.size());
  for (auto const t : tensors) {
    nodeouts.emplace_back(t);
  }
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Concat")
                  .Input(concat_dim)
                  .Input(nodeouts)
                  .Finalize(g, &ret));
  return ret;
}

Node* ConcatV2(Graph* g, gtl::ArraySlice<Node*> tensors, Node* concat_dim) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_37(mht_37_v, 669, "", "./tensorflow/core/graph/testlib.cc", "ConcatV2");

  std::vector<NodeBuilder::NodeOut> nodeouts;
  nodeouts.reserve(tensors.size());
  for (auto const t : tensors) {
    nodeouts.emplace_back(t);
  }
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "ConcatV2")
                  .Input(nodeouts)
                  .Input(concat_dim)
                  .Finalize(g, &ret));
  return ret;
}

Node* Next(Graph* g, const string& name, Node* input) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_38(mht_38_v, 687, "", "./tensorflow/core/graph/testlib.cc", "Next");

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(name, "NextIteration").Input(input).Finalize(g, &ret));
  return ret;
}

Node* LoopCond(Graph* g, Node* input) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_39(mht_39_v, 697, "", "./tensorflow/core/graph/testlib.cc", "LoopCond");

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("n"), "LoopCond").Input(input).Finalize(g, &ret));
  return ret;
}

Node* Less(Graph* g, Node* in0, Node* in1) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_40(mht_40_v, 707, "", "./tensorflow/core/graph/testlib.cc", "Less");

  return Binary(g, "Less", in0, in1);
}

Node* Select(Graph* g, Node* c, Node* inx, Node* iny) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_41(mht_41_v, 714, "", "./tensorflow/core/graph/testlib.cc", "Select");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Select")
                  .Input(c)
                  .Input(inx)
                  .Input(iny)
                  .Finalize(g, &ret));
  return ret;
}

Node* Cast(Graph* g, Node* in, DataType dst) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_42(mht_42_v, 727, "", "./tensorflow/core/graph/testlib.cc", "Cast");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Cast")
                  .Input(in)
                  .Attr("DstT", dst)
                  .Finalize(g, &ret));
  return ret;
}

Node* Gather(Graph* g, Node* in0, Node* in1, Node* axis) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_43(mht_43_v, 739, "", "./tensorflow/core/graph/testlib.cc", "Gather");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "GatherV2")
                  .Input(in0)
                  .Input(in1)
                  .Input(axis)
                  .Finalize(g, &ret));
  return ret;
}

Node* GetSessionTensor(Graph* g, Node* in) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_44(mht_44_v, 752, "", "./tensorflow/core/graph/testlib.cc", "GetSessionTensor");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "GetSessionTensor")
                  .Input(in, 0)
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(g, &ret));
  return ret;
}

Node* Relu(Graph* g, Node* in) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_45(mht_45_v, 764, "", "./tensorflow/core/graph/testlib.cc", "Relu");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Relu")
                  .Input(in, 0)
                  .Attr("T", DT_FLOAT)
                  .Finalize(g, &ret));
  return ret;
}

Node* Relu6(Graph* g, Node* in) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_46(mht_46_v, 776, "", "./tensorflow/core/graph/testlib.cc", "Relu6");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Relu6")
                  .Input(in, 0)
                  .Attr("T", DT_FLOAT)
                  .Finalize(g, &ret));
  return ret;
}

Node* BiasAdd(Graph* g, Node* value, Node* bias) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_47(mht_47_v, 788, "", "./tensorflow/core/graph/testlib.cc", "BiasAdd");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BiasAdd")
                  .Input(value)
                  .Input(bias)
                  .Attr("T", DT_FLOAT)
                  .Finalize(g, &ret));
  return ret;
}

Node* Conv2D(Graph* g, Node* in0, Node* in1) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_48(mht_48_v, 801, "", "./tensorflow/core/graph/testlib.cc", "Conv2D");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Conv2D")
                  .Input(in0)
                  .Input(in1)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Finalize(g, &ret));
  return ret;
}

Node* Diag(Graph* g, Node* in, DataType type) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_49(mht_49_v, 816, "", "./tensorflow/core/graph/testlib.cc", "Diag");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Diag")
                  .Input(in)
                  .Attr("T", type)
                  .Finalize(g, &ret));
  return ret;
}

Node* DiagPart(Graph* g, Node* in, DataType type) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_50(mht_50_v, 828, "", "./tensorflow/core/graph/testlib.cc", "DiagPart");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "DiagPart")
                  .Input(in)
                  .Attr("T", type)
                  .Finalize(g, &ret));
  return ret;
}

Node* CheckNumerics(Graph* g, Node* in, const string& message) {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("message: \"" + message + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_51(mht_51_v, 841, "", "./tensorflow/core/graph/testlib.cc", "CheckNumerics");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "CheckNumerics")
                  .Input(in)
                  .Attr("message", message)
                  .Finalize(g, &ret));
  return ret;
}

Node* Arg(Graph* g, int64_t index, DataType type) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_52(mht_52_v, 853, "", "./tensorflow/core/graph/testlib.cc", "Arg");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_Arg")
                  .Attr("T", type)
                  .Attr("index", index)
                  .Finalize(g, &ret));
  return ret;
}

Node* Retval(Graph* g, int64_t index, Node* in, int64_t in_index) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_53(mht_53_v, 865, "", "./tensorflow/core/graph/testlib.cc", "Retval");

  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "_Retval")
                  .Input(in, in_index)
                  .Attr("index", index)
                  .Finalize(g, &ret));
  return ret;
}

void ToGraphDef(Graph* g, GraphDef* gdef) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSgraphPStestlibDTcc mht_54(mht_54_v, 877, "", "./tensorflow/core/graph/testlib.cc", "ToGraphDef");
 g->ToGraphDef(gdef); }

}  // end namespace graph
}  // end namespace test
}  // end namespace tensorflow
