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

#ifndef TENSORFLOW_CORE_KERNELS_CONTROL_FLOW_OPS_H_
#define TENSORFLOW_CORE_KERNELS_CONTROL_FLOW_OPS_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh() {
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


#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

// A ControlTriggerOp is similar to a NoOp. However, it always treats the input
// control edges as Live edges. Its primary use so far is in the scheduling of
// recvs, where we add ControlTrigger nodes and use them to trigger recvs. We
// allow ControlTrigger nodes to be enabled by dead nodes.
class ControlTriggerOp : public OpKernel {
 public:
  explicit ControlTriggerOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh mht_0(mht_0_v, 200, "", "./tensorflow/core/kernels/control_flow_ops.h", "Compute");
}
  bool IsExpensive() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh mht_1(mht_1_v, 204, "", "./tensorflow/core/kernels/control_flow_ops.h", "IsExpensive");
 return false; }
};

// A switch op has two inputs and two outputs. It forwards the value of
// Input:0 to the output specified by input:1. Input:1 is a boolean tensor.
// Input:0 is forwarded to output:0 if input:1 is false, otherwise to
// output:1.
class SwitchOp : public OpKernel {
 public:
  explicit SwitchOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh mht_2(mht_2_v, 218, "", "./tensorflow/core/kernels/control_flow_ops.h", "IsExpensive");
 return false; }
  ~SwitchOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(SwitchOp);
};

// An n-way switch op has two inputs and N outputs. It forwards the value of
// Input:0 to the output specified by Input:1. Input:1 is an integer tensor.
// Input:0 is forwarded to output:0 if Input:1 is 0, to output:1 if 1, and so
// forth. If Input:1 is <0 or >=num_outputs(), Input:0 is forwarded to
// output:num_outputs()-1.
class SwitchNOp : public OpKernel {
 public:
  explicit SwitchNOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh mht_3(mht_3_v, 236, "", "./tensorflow/core/kernels/control_flow_ops.h", "IsExpensive");
 return false; }
  ~SwitchNOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(SwitchNOp);
};

// A merge op has n inputs and two outputs. It forwards the value of the
// first input that becomes available to its first output, and the
// index of the first input to its second output.
class MergeOp : public OpKernel {
 public:
  explicit MergeOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh mht_4(mht_4_v, 252, "", "./tensorflow/core/kernels/control_flow_ops.h", "IsExpensive");
 return false; }
  ~MergeOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(MergeOp);
};

// An enter op has one input and one output. It creates or finds
// the child frame that is uniquely identified by the frame_name,
// and makes its input available to the child frame.
class EnterOp : public OpKernel {
 public:
  explicit EnterOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh mht_5(mht_5_v, 268, "", "./tensorflow/core/kernels/control_flow_ops.h", "IsExpensive");
 return false; }
  ~EnterOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(EnterOp);
};

// An exit op has one input and one output. It exits the current
// frame to its parent frame, and makes its input available to the
// parent frame.
class ExitOp : public OpKernel {
 public:
  explicit ExitOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh mht_6(mht_6_v, 284, "", "./tensorflow/core/kernels/control_flow_ops.h", "IsExpensive");
 return false; }
  ~ExitOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(ExitOp);
};

// A next_iteration op has one input and one output. It makes its input
// available to the next iteration.
class NextIterationOp : public OpKernel {
 public:
  explicit NextIterationOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override;
  bool IsExpensive() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScontrol_flow_opsDTh mht_7(mht_7_v, 299, "", "./tensorflow/core/kernels/control_flow_ops.h", "IsExpensive");
 return false; }
  ~NextIterationOp() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(NextIterationOp);
};

// A LoopCond op has one input and one output. The input is a boolean
// scalar representing the taken branches of the "pivot" Switch that
// determines loop termination. As a contract, any high-level front-end
// should always use port '0' of the "pivot" switches for loop exit.
class LoopCondOp : public OpKernel {
 public:
  explicit LoopCondOp(OpKernelConstruction* context);
  ~LoopCondOp() override;

  void Compute(OpKernelContext* context) override;

  bool IsExpensive() override;

  TF_DISALLOW_COPY_AND_ASSIGN(LoopCondOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONTROL_FLOW_OPS_H_
