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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_manager_testDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_manager_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_manager_testDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {

using ::testing::UnorderedElementsAre;

class XlaDebugInfoManagerTest : public HloTestBase {
 protected:
  struct DebugMetadata {
    // We allow same id to be registered multiple times. we need unique id to
    // know which program is referenced (such as in UnregisterProgram).
    int unique_id;
    std::string id;
    std::shared_ptr<HloModule> module;
    std::shared_ptr<BufferAssignmentProto> buffer_assignment;
  };

  // Return unique id of this module.
  int RegisterProgram(const std::string& module_id) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("module_id: \"" + module_id + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_manager_testDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager_test.cc", "RegisterProgram");

    DebugMetadata debug_info;
    HloModuleConfig config;
    debug_info.unique_id = ++serial_;
    debug_info.id = module_id;
    debug_info.module = std::make_shared<HloModule>(module_id, config);
    debug_info.buffer_assignment = nullptr;
    xla_debug_info_manager_.RegisterModule(module_id, debug_info.module,
                                           debug_info.buffer_assignment);
    external_references_.push_back(std::move(debug_info));
    return serial_;
  }

  void UnregisterProgram(int unique_id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_manager_testDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager_test.cc", "UnregisterProgram");

    for (int i = 0; i < external_references_.size(); i++) {
      if (external_references_[i].unique_id == unique_id) {
        xla_debug_info_manager_.UnregisterModule(
            external_references_[i].id, external_references_[i].module,
            external_references_[i].buffer_assignment);
        external_references_.erase(external_references_.begin() + i);
        break;
      }
    }
  }

  void StartProgram(int unique_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_manager_testDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager_test.cc", "StartProgram");

    for (int i = 0; i < external_references_.size(); i++) {
      if (external_references_[i].unique_id == unique_id) {
        xla_debug_info_manager_.OnModuleStart(external_references_[i].id);
        break;
      }
    }
  }

  void StopProgram(int unique_id) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_manager_testDTcc mht_3(mht_3_v, 249, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager_test.cc", "StopProgram");

    for (int i = 0; i < external_references_.size(); i++) {
      if (external_references_[i].unique_id == unique_id) {
        xla_debug_info_manager_.OnModuleStop(external_references_[i].id);
        break;
      }
    }
  }

  void StartAndStopProgram(int unique_id) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_manager_testDTcc mht_4(mht_4_v, 261, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager_test.cc", "StartAndStopProgram");

    StartProgram(unique_id);
    StopProgram(unique_id);
  }

  std::set<ModuleIdentifier> GetActiveModule() {
    return xla_debug_info_manager_.GetActiveModules();
  }

  void StartTrace() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSxla_debug_info_manager_testDTcc mht_5(mht_5_v, 273, "", "./tensorflow/compiler/xla/service/xla_debug_info_manager_test.cc", "StartTrace");
 xla_debug_info_manager_.StartTracing(); }

  std::set<ModuleIdentifier> StopTrace() {
    std::vector<XlaModuleDebugInfo> module_debug_info;
    xla_debug_info_manager_.StopTracing(&module_debug_info);
    std::set<ModuleIdentifier> serialized;
    for (const auto& module : module_debug_info) {
      serialized.insert(module.module_id);
    }
    return serialized;
  }

  int serial_ = 0;

  // Simulation of compilation cache.
  std::vector<DebugMetadata> external_references_;

  // Use an instance per test instead of singleton to avoid interferences.
  XlaDebugInfoManager xla_debug_info_manager_;
};

// Test the cases where no trace session is involved.
TEST_F(XlaDebugInfoManagerTest, NoTraceBasic) {
  auto program0 = RegisterProgram("program0");
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));

  auto program1 = RegisterProgram("program1");
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));

  StartAndStopProgram(program0);
  StartProgram(program0);
  StopProgram(program0);
  UnregisterProgram(program0);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program1"));
  StartAndStopProgram(program1);
  StartProgram(program1);
  StopProgram(program1);
  UnregisterProgram(program1);
  EXPECT_TRUE(GetActiveModule().empty());
}

TEST_F(XlaDebugInfoManagerTest, NoTraceDuplicateIds) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));

  StartProgram(program0A);
  StartProgram(program0B);
  StartProgram(program1);
  StopProgram(program0A);
  StopProgram(program0B);
  StopProgram(program1);

  UnregisterProgram(program1);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));
  UnregisterProgram(program0A);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));
  UnregisterProgram(program0B);
  EXPECT_TRUE(GetActiveModule().empty());
}

// Test the cases where an active trace session is involved.
TEST_F(XlaDebugInfoManagerTest, ActiveTrace) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");

  // Case 1: Trace starts when no program is running.
  StartAndStopProgram(program0A);
  StartTrace();
  StartAndStopProgram(program1);
  auto program2 = RegisterProgram("program2");
  StartAndStopProgram(program0B);
  EXPECT_THAT(StopTrace(),
              UnorderedElementsAre("program0", "program1", "program2"));

  // Case 1: Trace starts during program is running.
  StartProgram(program0A);
  StartTrace();
  StopProgram(program0A);
  StartAndStopProgram(program1);
  EXPECT_THAT(StopTrace(),
              UnorderedElementsAre("program0", "program1", "program2"));

  UnregisterProgram(program2);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));
  UnregisterProgram(program0A);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0", "program1"));
  UnregisterProgram(program0B);
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program1"));
  UnregisterProgram(program1);
  EXPECT_TRUE(GetActiveModule().empty());
}

TEST_F(XlaDebugInfoManagerTest, UnregisterDuringTrace) {
  auto program0A = RegisterProgram("program0");
  auto program0B = RegisterProgram("program0");  // duplicates
  auto program1 = RegisterProgram("program1");

  StartTrace();
  StartAndStopProgram(program1);
  UnregisterProgram(program1);
  UnregisterProgram(program0B);
  EXPECT_THAT(StopTrace(), UnorderedElementsAre("program0", "program1"));
  EXPECT_THAT(GetActiveModule(), UnorderedElementsAre("program0"));

  UnregisterProgram(program0A);
}

}  // namespace xla
