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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc() {
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
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"

#include <algorithm>
#include <cctype>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
struct RegisteredTargetHardware {
  // TODO(b/177376459): Remove this constructor.
  RegisteredTargetHardware(const std::string& name,
                           const std::string& description, mlir::TypeID type_id,
                           std::unique_ptr<TargetHardware> target_hardware)
      : unique_name(GetCanonicalHardwareName(name)),
        description(description),
        type_id(type_id),
        target_hardware(std::move(target_hardware)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   mht_0_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "RegisteredTargetHardware");
}

  RegisteredTargetHardware(
      const std::string& name, const std::string& description,
      mlir::TypeID type_id,
      std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory)
      : unique_name(GetCanonicalHardwareName(name)),
        description(description),
        target_hardware_factory(target_hardware_factory) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   mht_1_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_1(mht_1_v, 225, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "RegisteredTargetHardware");
}

  std::string unique_name;
  std::string description;
  mlir::TypeID type_id;
  std::unique_ptr<TargetHardware> target_hardware;
  std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory;
};

struct RegisteredTargetHardwareOps {
  explicit RegisteredTargetHardwareOps(mlir::TypeID hardware_type)
      : hardware_typeid(hardware_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_2(mht_2_v, 239, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "RegisteredTargetHardwareOps");
}
  // Key is the Operation TypeID
  llvm::DenseMap<mlir::TypeID, std::unique_ptr<TargetHardwareOperation>>
      target_hardware_ops;
  // Key is the Operation TypeID
  llvm::DenseMap<mlir::TypeID,
                 std::function<std::unique_ptr<TargetHardwareOperation>()>>
      target_hardware_ops_factory;
  mlir::TypeID hardware_typeid;
};

std::vector<std::unique_ptr<RegisteredTargetHardwareOps>>*
GetRegisteredTargetHardwareOps() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_3(mht_3_v, 254, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "GetRegisteredTargetHardwareOps");

  static std::vector<std::unique_ptr<RegisteredTargetHardwareOps>>*
      hardwares_ops =
          []() -> std::vector<std::unique_ptr<RegisteredTargetHardwareOps>>* {
    return new std::vector<std::unique_ptr<RegisteredTargetHardwareOps>>();
  }();
  return hardwares_ops;
}

std::vector<RegisteredTargetHardware>* GetRegisteredHardwares() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_4(mht_4_v, 266, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "GetRegisteredHardwares");

  static std::vector<RegisteredTargetHardware>* hardwares =
      []() -> std::vector<RegisteredTargetHardware>* {
    return new std::vector<RegisteredTargetHardware>();
  }();
  return hardwares;
}

llvm::DenseMap<mlir::TypeID, std::unique_ptr<TargetHardwareOperation>>*
getRegisteredOperationsForHardware(mlir::TypeID type_id) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_5(mht_5_v, 278, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "getRegisteredOperationsForHardware");

  auto* hardwares = GetRegisteredTargetHardwareOps();
  for (auto& hardware : *hardwares) {
    if (hardware->hardware_typeid == type_id) {
      return &hardware->target_hardware_ops;
    }
  }
  return nullptr;
}

// A deny list for op cost computation since those ops are not arithemtic.
inline bool IsNonArithmeticOp(mlir::Operation* op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_6(mht_6_v, 292, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "IsNonArithmeticOp");

  if (llvm::isa<func::ReturnOp, FuncOp>(op)) return true;
  if (op->hasTrait<OpTrait::ConstantLike>()) return true;
  if (llvm::isa<QConstOp, SparseQConstOp>(op)) return true;
  if (!NotTFLQuantDequantizeOp(op)) return true;
  return false;
}

}  // namespace

bool TargetHardware::Init() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_7(mht_7_v, 305, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "TargetHardware::Init");

  auto* hardware_ops_factory = GetRegisteredTargetHardwareOps();
  for (auto& hardware_ops : *hardware_ops_factory) {
    if (hardware_ops->hardware_typeid != this->GetTypeId()) continue;
    auto& op_factories = hardware_ops->target_hardware_ops_factory;
    for (auto& op_factory : op_factories) {
      hardware_ops_.emplace_back(op_factory.getSecond()());
    }
    break;
  }
  return true;
}

double TargetHardware::GetOpCost(mlir::Operation* op) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_8(mht_8_v, 321, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "TargetHardware::GetOpCost");

  auto* registered_ops = getRegisteredOperationsForHardware(GetTypeId());
  if (registered_ops == nullptr) {
    return kDefaultFixedValuedCost;
  }
  auto abstract_op = op->getRegisteredInfo();
  auto hardware_op = registered_ops->find(abstract_op->getTypeID());
  if (hardware_op == registered_ops->end()) return kDefaultFixedValuedCost;
  return hardware_op->second->GetOpCost(op);
}

bool TargetHardware::IsOpSupported(mlir::Operation* op) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_9(mht_9_v, 335, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "TargetHardware::IsOpSupported");

  auto* registered_ops = getRegisteredOperationsForHardware(GetTypeId());
  if (registered_ops == nullptr) {
    return false;
  }
  auto abstract_op = op->getRegisteredInfo();
  auto hardware_op = registered_ops->find(abstract_op->getTypeID());
  if (hardware_op == registered_ops->end()) return false;
  return hardware_op->second->IsOpSupported(op);
}

double TargetHardware::GetFuncCost(FuncOp* func) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_10(mht_10_v, 349, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "TargetHardware::GetFuncCost");

  double total_cost = 0.0;
  func->walk([&](Operation* op) {
    if (IsNonArithmeticOp(op)) return;
    // We will always defer to the hardware to decide the cost.
    total_cost += GetOpCost(op);
  });
  return total_cost;
}

const TargetHardware* GetTargetHardware(const std::string& hardware_name) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("hardware_name: \"" + hardware_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_11(mht_11_v, 363, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "GetTargetHardware");

  const std::string canonical_name = GetCanonicalHardwareName(hardware_name);
  // Just loop for now, we don't expect number of hardwares to be huge.
  // Revisit to have map if number of elements increased.
  auto* registered_hardwares = GetRegisteredHardwares();
  for (const auto& hardware : *registered_hardwares) {
    if (hardware.unique_name == canonical_name) {
      return hardware.target_hardware.get();
    }
  }
  return nullptr;
}

std::function<std::unique_ptr<TargetHardware>()> GetTargetHardwareFactory(
    const std::string& hardware_name) {
  const std::string canonical_name = GetCanonicalHardwareName(hardware_name);
  // Just loop for now, we don't expect number of hardwares to be huge.
  // Revisit to have map if number of elements increased.
  auto* registered_hardwares = GetRegisteredHardwares();
  for (const auto& hardware : *registered_hardwares) {
    if (hardware.unique_name == canonical_name) {
      return hardware.target_hardware_factory;
    }
  }
  return nullptr;
}

namespace internal {

void RegisterTargetHardware(
    const std::string& unique_name, const std::string& description,
    mlir::TypeID type_id,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("unique_name: \"" + unique_name + "\"");
   mht_12_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_12(mht_12_v, 400, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "RegisterTargetHardware");

  auto* registered_hardwares = GetRegisteredHardwares();
  for (const auto& hardware : *registered_hardwares) {
    if (hardware.unique_name == unique_name) {
      llvm::errs() << "Ignoring duplicate hardware. Hardware " << unique_name
                   << " already registered\n";
      return;
    }
  }
  registered_hardwares->push_back(RegisteredTargetHardware(
      unique_name, description, type_id, target_hardware_factory()));
}

void RegisterTargetHardwareFactory(
    const std::string& unique_name, const std::string& description,
    mlir::TypeID type_id,
    std::function<std::unique_ptr<TargetHardware>()> target_hardware_factory) {
   std::vector<std::string> mht_13_v;
   mht_13_v.push_back("unique_name: \"" + unique_name + "\"");
   mht_13_v.push_back("description: \"" + description + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_13(mht_13_v, 421, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "RegisterTargetHardwareFactory");

  auto* registered_hardwares = GetRegisteredHardwares();
  for (auto& hardware : *registered_hardwares) {
    if (hardware.unique_name == unique_name) {
      llvm::errs() << "Ignoring duplicate hardware. Hardware " << unique_name
                   << " already registered\n";
      hardware.target_hardware_factory = target_hardware_factory;
      return;
    }
  }
  registered_hardwares->push_back(RegisteredTargetHardware(
      unique_name, description, type_id, target_hardware_factory));
}

void RegisterTargetHardwareOp(
    mlir::TypeID hardware_type, mlir::TypeID op_type,
    std::function<std::unique_ptr<TargetHardwareOperation>()>
        target_hardware_op_factory) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_14(mht_14_v, 441, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "RegisterTargetHardwareOp");

  auto* registered_hardware_ops = GetRegisteredTargetHardwareOps();
  for (auto& hardware : *registered_hardware_ops) {
    if (hardware->hardware_typeid == hardware_type) {
      if (hardware->target_hardware_ops.count(op_type)) {
        llvm::errs() << "Trying to register duplicate Op";
        return;
      }
      hardware->target_hardware_ops[op_type] = target_hardware_op_factory();
      return;
    }
  }
  registered_hardware_ops->push_back(
      std::make_unique<RegisteredTargetHardwareOps>(
          RegisteredTargetHardwareOps(hardware_type)));
  registered_hardware_ops->back()->target_hardware_ops[op_type] =
      target_hardware_op_factory();
}

void RegisterTargetHardwareOpFactory(
    mlir::TypeID hardware_type, mlir::TypeID op_type,
    std::function<std::unique_ptr<TargetHardwareOperation>()>
        target_hardware_op_factory) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_15(mht_15_v, 466, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "RegisterTargetHardwareOpFactory");

  auto* registered_hardware_ops = GetRegisteredTargetHardwareOps();
  for (auto& hardware : *registered_hardware_ops) {
    if (hardware->hardware_typeid == hardware_type) {
      if (hardware->target_hardware_ops_factory.count(op_type)) {
        llvm::errs() << "Trying to register duplicate Op";
        return;
      }
      hardware->target_hardware_ops_factory[op_type] =
          target_hardware_op_factory;
      return;
    }
  }
  registered_hardware_ops->push_back(
      std::make_unique<RegisteredTargetHardwareOps>(
          RegisteredTargetHardwareOps(hardware_type)));
  registered_hardware_ops->back()->target_hardware_ops_factory[op_type] =
      target_hardware_op_factory;
}

}  // namespace internal

bool ProcessTargetDevices(llvm::ArrayRef<std::string> specified_device_specs,
                          std::vector<std::string>* device_specs) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_16(mht_16_v, 492, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "ProcessTargetDevices");

  bool cpu_include = false;
  for (auto& device_spec : specified_device_specs) {
    auto device = GetCanonicalHardwareName(device_spec);

    if (device == "CPU") cpu_include = true;
    device_specs->push_back(device);
  }
  if (!cpu_include) {
    device_specs->push_back("CPU");
  }

  // Make sure all the devices are registered.
  for (const std::string& device : *device_specs) {
    if (GetTargetHardware(device) == nullptr) {
      llvm::errs() << "cannot get target hardware for device: " << device;
      return false;
    }
  }

  return true;
}

std::string GetHardwareName(const TargetHardware* hardware) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSexperimentalPStacPShardwaresPStarget_hardwareDTcc mht_17(mht_17_v, 518, "", "./tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.cc", "GetHardwareName");

  const auto* registered_hardwares = GetRegisteredHardwares();
  for (const auto& registered_hardware : *registered_hardwares) {
    if (registered_hardware.type_id == hardware->GetTypeId())
      return registered_hardware.unique_name;
  }
  return "";
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
