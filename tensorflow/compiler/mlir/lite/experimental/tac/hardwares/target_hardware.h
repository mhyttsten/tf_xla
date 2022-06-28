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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TARGET_HARDWARE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TARGET_HARDWARE_H_
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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh() {
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


#include <memory>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace tac {

// Default fixed values for ops.
constexpr static float kDefaultFixedValuedCost = 1000000.0;

// This is just fake data.
constexpr static float kCrossHardwareTransferPerByteCost = 5.0f;

// This is just fake data.
constexpr static float kCrossHardwareTransferFixedCost = 10.f;

// Interface for an Operation capabilities which should be tied to
// a specific hardware.
// Users should implement the interface and use TargetHardwareOpRegistration
// for registering the operation.
class TargetHardwareOperation {
 public:
  virtual ~TargetHardwareOperation() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h", "~TargetHardwareOperation");
}

  virtual double GetOpCost(mlir::Operation* op) const = 0;

  virtual bool IsOpSupported(mlir::Operation* op) const = 0;
};

// Abstract base class for a hardware.
// To introduce new hardware
// users should implement the interface and use TargetHardwareRegistration
// for registering the hardware.
// Subclasses must implement the pure virtual function interface and
// define static member variable that retrieves string identifying the Target
// Hardware. Example,
// class MyType : public TargetHardware {
//  public:
//   static constexpr char kId[] = "MyHardware";
// };
class TargetHardware {
 public:
  virtual ~TargetHardware() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh mht_1(mht_1_v, 237, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h", "~TargetHardware");
}

  // Initializes all TargetHardwareOperation registered for this hardware.
  // Users overriding this function, should call the base class method to
  // initialize the ops.
  virtual bool Init();

  // Returns the cost of running 'op' on this Hardware.
  virtual double GetOpCost(mlir::Operation* op) const;

  // Returns the cost of running the whole function on this hardware.
  // By default this is the sum of the cost of individual cost for each op.
  virtual double GetFuncCost(FuncOp* func) const;

  // Returns true if 'op' can run on this Hardware.
  virtual bool IsOpSupported(mlir::Operation* op) const;

  // Switching cost between from hardware and this hardware.
  // If both the hardwares are the same, the transfer cost is basically 0.
  virtual double GetHardwareSwitchingCost(const TargetHardware* from,
                                          size_t buffer_size) const = 0;

  // Returns a list of all patterns to apply for this hardware.
  virtual mlir::RewritePatternSet GetTransformations(
      MLIRContext* context) const = 0;

  // Returns TypeId for the provided hardware.
  // Usually should be something like mlir::TypeID::get<MyType>()
  virtual mlir::TypeID GetTypeId() const = 0;

 protected:
  // All registered hardware ops.
  std::vector<std::unique_ptr<TargetHardwareOperation>> hardware_ops_;
};

// Returns pointer to the Hardware identified by 'hardware_name'.
// If not found nullptr is returned.
// DEPRECATED: Do not use, prefer GetTargetHardwareFactory instead.
const TargetHardware* GetTargetHardware(const std::string& hardware_name);

// Returns the factory method for the requested hardware if present.
std::function<std::unique_ptr<TargetHardware>()> GetTargetHardwareFactory(
    const std::string& hardware_name);

namespace internal {
// DEPRECATED: Do not use, prefer using RegisterTargetHardwareFactory instead.
void RegisterTargetHardware(
    const std::string& unique_name, const std::string& description,
    mlir::TypeID type_id,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory);

// DEPRECATED: Do not use, prefer using RegisterTargetHardwareFactory instead.
template <typename T>
void RegisterTargetHardware(
    const std::string& description,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh mht_2(mht_2_v, 296, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h", "RegisterTargetHardware");

  RegisterTargetHardware(T::kId, description, mlir::TypeID::get<T>(),
                         target_hardware_factory);
}

void RegisterTargetHardwareFactory(
    const std::string& unique_name, const std::string& description,
    mlir::TypeID type_id,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory);

// Registers the provided target hardware factory.
template <typename T>
void RegisterTargetHardwareFactory(
    const std::string& description,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh mht_3(mht_3_v, 314, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h", "RegisterTargetHardwareFactory");

  RegisterTargetHardwareFactory(T::kId, description, mlir::TypeID::get<T>(),
                                target_hardware_factory);
}

// DEPRECATED: Do not use, prefer RegisterTargetHardwareOpFactory intstead.
void RegisterTargetHardwareOp(
    mlir::TypeID hardware_type, mlir::TypeID op_type,
    std::function<std::unique_ptr<TargetHardwareOperation>()>
        target_hardware_op_factory);

void RegisterTargetHardwareOpFactory(
    mlir::TypeID hardware_type, mlir::TypeID op_type,
    std::function<std::unique_ptr<TargetHardwareOperation>()>
        target_hardware_op_factory);
}  // namespace internal

// Register target hardware.
template <typename Hardware>
struct TargetHardwareRegistration {
  TargetHardwareRegistration(const std::string& description,
                             std::function<std::unique_ptr<TargetHardware>()>
                                 target_hardware_factory) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh mht_4(mht_4_v, 340, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h", "TargetHardwareRegistration");

    // TODO(b/177376459): remove this.
    internal::RegisterTargetHardware<Hardware>(description,
                                               target_hardware_factory);
    internal::RegisterTargetHardwareFactory<Hardware>(description,
                                                      target_hardware_factory);
  }
};

// Register Op capabilities for specific hardware.
template <typename Hardware, typename Op>
struct TargetHardwareOpRegistration {
  explicit TargetHardwareOpRegistration(
      std::function<std::unique_ptr<TargetHardwareOperation>()>
          target_hardware_op_factory) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh mht_5(mht_5_v, 357, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h", "TargetHardwareOpRegistration");

    // TODO(b/177376459): remove this.
    internal::RegisterTargetHardwareOp(mlir::TypeID::get<Hardware>(),
                                       mlir::TypeID::get<Op>(),
                                       target_hardware_op_factory);
    internal::RegisterTargetHardwareOpFactory(mlir::TypeID::get<Hardware>(),
                                              mlir::TypeID::get<Op>(),
                                              target_hardware_op_factory);
  }
};

//======== util functions ==========

// Process user specified device specs, will always add CPU if it's not there.
// specified_deivce_specs: ',' separated, like "GPU,DSP,CPU".
// device_specs: processed device specs enum.
bool ProcessTargetDevices(llvm::ArrayRef<std::string> specified_device_specs,
                          std::vector<std::string>* device_specs);

// Check whether two hardwares are the same.
inline bool IsSameHardware(const TargetHardware* lhs,
                           const TargetHardware* rhs) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTh mht_6(mht_6_v, 381, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h", "IsSameHardware");

  return lhs->GetTypeId() == rhs->GetTypeId();
}

// Returns the ID identifying 'hardware'. This should match the ID defined
// in the hardware field ID.
// For example, if MyHardware is passed the value returned should match
// MyHardware::kId.
std::string GetHardwareName(const TargetHardware* hardware);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TARGET_HARDWARE_H_
