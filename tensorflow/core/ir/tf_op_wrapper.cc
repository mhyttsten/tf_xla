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
class MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc {
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
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/tf_op_wrapper.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"

namespace mlir {
namespace tfg {

TFOp::TFOp(Operation *op) : op_(op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_0(mht_0_v, 193, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::TFOp");

  assert(!op || classof(op) && "expected a TFG op");
}

StringAttr TFOp::nameAttr() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_1(mht_1_v, 200, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::nameAttr");

  return op_->getAttrOfType<StringAttr>(getDialect()->getNameAttrIdentifier());
}

StringRef TFOp::name() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_2(mht_2_v, 207, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::name");
 return nameAttr().getValue(); }

void TFOp::setName(const Twine &name) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_3(mht_3_v, 212, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::setName");

  setName(StringAttr::get(op_->getContext(), name.str()));
}

void TFOp::setName(StringAttr name) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_4(mht_4_v, 219, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::setName");

  op_->setAttr(getDialect()->getNameAttrIdentifier(), name);
}

StringAttr TFOp::requestedDeviceAttr() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_5(mht_5_v, 226, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::requestedDeviceAttr");

  return op_->getAttrOfType<StringAttr>(
      getDialect()->getDeviceAttrIdentifier());
}

StringRef TFOp::requestedDevice() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_6(mht_6_v, 234, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::requestedDevice");
 return requestedDeviceAttr().getValue(); }

void TFOp::setRequestedDevice(const Twine &device) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_7(mht_7_v, 239, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::setRequestedDevice");

  setRequestedDevice(StringAttr::get(op_->getContext(), device.str()));
}

void TFOp::setRequestedDevice(StringAttr device) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_8(mht_8_v, 246, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::setRequestedDevice");

  op_->setAttr(getDialect()->getDeviceAttrIdentifier(), device);
}

StringAttr TFOp::assignedDeviceAttr() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_9(mht_9_v, 253, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::assignedDeviceAttr");

  return op_->getAttrOfType<StringAttr>(
      getDialect()->getAssignedDeviceAttrIdentifier());
}

StringRef TFOp::assignedDevice() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_10(mht_10_v, 261, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::assignedDevice");
 return assignedDeviceAttr().getValue(); }

void TFOp::setAssignedDevice(const Twine &device) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_11(mht_11_v, 266, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::setAssignedDevice");

  setAssignedDevice(StringAttr::get(op_->getContext(), device.str()));
}

void TFOp::setAssignedDevice(StringAttr device) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_12(mht_12_v, 273, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::setAssignedDevice");

  op_->setAttr(getDialect()->getAssignedDeviceAttrIdentifier(), device);
}

StringAttr TFOp::tpuReplicate() {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_13(mht_13_v, 280, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::tpuReplicate");

  return op_->getAttrOfType<StringAttr>(
      getDialect()->getTfgTpuReplicateAttrIdentifier());
}

void TFOp::setTpuReplicate(StringAttr tpu_replicate) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSirPStf_op_wrapperDTcc mht_14(mht_14_v, 288, "", "./tensorflow/core/ir/tf_op_wrapper.cc", "TFOp::setTpuReplicate");

  op_->setAttr(getDialect()->getTfgTpuReplicateAttrIdentifier(), tpu_replicate);
}

}  // namespace tfg
}  // namespace mlir
