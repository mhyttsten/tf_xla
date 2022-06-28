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
class MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc {
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
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc() {
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

#include "tensorflow/c/eager/c_api_experimental.h"

#include <vector>

#include "absl/strings/match.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_op_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/eager_operation.h"
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_agent.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/monitoring/sampler.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/strcat.h"

using tensorflow::string;

void TFE_OpReset(TFE_Op* op_to_reset, const char* op_or_function_name,
                 const char* raw_device_name, TF_Status* status) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("op_or_function_name: \"" + (op_or_function_name == nullptr ? std::string("nullptr") : std::string((char*)op_or_function_name)) + "\"");
   mht_0_v.push_back("raw_device_name: \"" + (raw_device_name == nullptr ? std::string("nullptr") : std::string((char*)raw_device_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_0(mht_0_v, 213, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_OpReset");

  if (op_to_reset) {
    tensorflow::ImmediateExecutionOperation* op =
        tensorflow::unwrap(op_to_reset);
    op->Clear();
    status->status = op->Reset(op_or_function_name, raw_device_name);
  } else {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "op_to_reset should not be nullptr");
  }
}

void TFE_ContextEnableGraphCollection(TFE_Context* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_1(mht_1_v, 228, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextEnableGraphCollection");

  tensorflow::unwrap(ctx)->SetShouldStoreGraphs(true);
}

void TFE_ContextDisableGraphCollection(TFE_Context* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_2(mht_2_v, 235, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextDisableGraphCollection");

  tensorflow::unwrap(ctx)->SetShouldStoreGraphs(false);
}

uint64_t TFE_GetContextId(TFE_Context* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_3(mht_3_v, 242, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_GetContextId");

  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  return context->GetContextId();
}

void TFE_MonitoringCounterCellIncrementBy(TFE_MonitoringCounterCell* cell,
                                          int64_t value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_4(mht_4_v, 252, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringCounterCellIncrementBy");

  cell->cell.IncrementBy(value);
}

int64_t TFE_MonitoringCounterCellValue(TFE_MonitoringCounterCell* cell) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_5(mht_5_v, 259, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringCounterCellValue");

  return cell->cell.value();
}

TFE_MonitoringCounter0* TFE_MonitoringNewCounter0(const char* name,
                                                  TF_Status* status,
                                                  const char* description) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_6_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_6(mht_6_v, 270, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewCounter0");

  auto* result = new TFE_MonitoringCounter0({name, description});
  Set_TF_Status_from_Status(status, result->counter->GetStatus());
  if (!result->counter->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteCounter0(TFE_MonitoringCounter0* counter) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_7(mht_7_v, 283, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteCounter0");

  delete counter;
}

TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter0(
    TFE_MonitoringCounter0* counter) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_8(mht_8_v, 291, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellCounter0");

  return static_cast<TFE_MonitoringCounterCell*>(
      static_cast<void*>(counter->counter->GetCell()));
}

TFE_MonitoringCounter1* TFE_MonitoringNewCounter1(const char* name,
                                                  TF_Status* status,
                                                  const char* description,
                                                  const char* label1) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_9_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_9_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_9(mht_9_v, 305, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewCounter1");

  auto* result = new TFE_MonitoringCounter1({name, description, label1});
  Set_TF_Status_from_Status(status, result->counter->GetStatus());
  if (!result->counter->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteCounter1(TFE_MonitoringCounter1* counter) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_10(mht_10_v, 318, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteCounter1");

  delete counter;
}

TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter1(
    TFE_MonitoringCounter1* counter, const char* label1) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_11(mht_11_v, 327, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellCounter1");

  return static_cast<TFE_MonitoringCounterCell*>(
      static_cast<void*>(counter->counter->GetCell(label1)));
}

TFE_MonitoringCounter2* TFE_MonitoringNewCounter2(const char* name,
                                                  TF_Status* status,
                                                  const char* description,
                                                  const char* label1,
                                                  const char* label2) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_12_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_12_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_12_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_12(mht_12_v, 343, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewCounter2");

  auto* result =
      new TFE_MonitoringCounter2({name, description, label1, label2});
  Set_TF_Status_from_Status(status, result->counter->GetStatus());
  if (!result->counter->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteCounter2(TFE_MonitoringCounter2* counter) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_13(mht_13_v, 357, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteCounter2");

  delete counter;
}

TFE_MonitoringCounterCell* TFE_MonitoringGetCellCounter2(
    TFE_MonitoringCounter2* counter, const char* label1, const char* label2) {
   std::vector<std::string> mht_14_v;
   mht_14_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_14_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_14(mht_14_v, 367, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellCounter2");

  return static_cast<TFE_MonitoringCounterCell*>(
      static_cast<void*>(counter->counter->GetCell(label1, label2)));
}

void TFE_MonitoringIntGaugeCellSet(TFE_MonitoringIntGaugeCell* cell,
                                   int64_t value) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_15(mht_15_v, 376, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringIntGaugeCellSet");

  cell->cell.Set(value);
}

int64_t TFE_MonitoringIntGaugeCellValue(TFE_MonitoringIntGaugeCell* cell) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_16(mht_16_v, 383, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringIntGaugeCellValue");

  return cell->cell.value();
}

TFE_MonitoringIntGauge0* TFE_MonitoringNewIntGauge0(const char* name,
                                                    TF_Status* status,
                                                    const char* description) {
   std::vector<std::string> mht_17_v;
   mht_17_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_17_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_17(mht_17_v, 394, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewIntGauge0");

  auto* result = new TFE_MonitoringIntGauge0({name, description});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteIntGauge0(TFE_MonitoringIntGauge0* gauge) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_18(mht_18_v, 407, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteIntGauge0");

  delete gauge;
}

TFE_MonitoringIntGaugeCell* TFE_MonitoringGetCellIntGauge0(
    TFE_MonitoringIntGauge0* gauge) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_19(mht_19_v, 415, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellIntGauge0");

  return static_cast<TFE_MonitoringIntGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell()));
}

TFE_MonitoringIntGauge1* TFE_MonitoringNewIntGauge1(const char* name,
                                                    TF_Status* status,
                                                    const char* description,
                                                    const char* label1) {
   std::vector<std::string> mht_20_v;
   mht_20_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_20_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_20_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_20(mht_20_v, 429, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewIntGauge1");

  auto* result = new TFE_MonitoringIntGauge1({name, description, label1});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteIntGauge1(TFE_MonitoringIntGauge1* gauge) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_21(mht_21_v, 442, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteIntGauge1");

  delete gauge;
}

TFE_MonitoringIntGaugeCell* TFE_MonitoringGetCellIntGauge1(
    TFE_MonitoringIntGauge1* gauge, const char* label1) {
   std::vector<std::string> mht_22_v;
   mht_22_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_22(mht_22_v, 451, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellIntGauge1");

  return static_cast<TFE_MonitoringIntGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1)));
}

TFE_MonitoringIntGauge2* TFE_MonitoringNewIntGauge2(const char* name,
                                                    TF_Status* status,
                                                    const char* description,
                                                    const char* label1,
                                                    const char* label2) {
   std::vector<std::string> mht_23_v;
   mht_23_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_23_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_23_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_23_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_23(mht_23_v, 467, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewIntGauge2");

  auto* result =
      new TFE_MonitoringIntGauge2({name, description, label1, label2});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteIntGauge2(TFE_MonitoringIntGauge2* gauge) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_24(mht_24_v, 481, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteIntGauge2");

  delete gauge;
}

TFE_MonitoringIntGaugeCell* TFE_MonitoringGetCellIntGauge2(
    TFE_MonitoringIntGauge2* gauge, const char* label1, const char* label2) {
   std::vector<std::string> mht_25_v;
   mht_25_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_25_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_25(mht_25_v, 491, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellIntGauge2");

  return static_cast<TFE_MonitoringIntGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2)));
}

void TFE_MonitoringStringGaugeCellSet(TFE_MonitoringStringGaugeCell* cell,
                                      const char* value) {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_26(mht_26_v, 501, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringStringGaugeCellSet");

  cell->cell.Set({value});
}

const void TFE_MonitoringStringGaugeCellValue(
    TFE_MonitoringStringGaugeCell* cell, TF_Buffer* buf) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_27(mht_27_v, 509, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringStringGaugeCellValue");

  tensorflow::string value = cell->cell.value();
  void* data = tensorflow::port::Malloc(value.length());
  value.copy(static_cast<char*>(data), value.length(), 0);
  buf->data = data;
  buf->length = value.length();
  buf->data_deallocator = [](void* data, size_t length) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_28(mht_28_v, 518, "", "./tensorflow/c/eager/c_api_experimental.cc", "lambda");

    tensorflow::port::Free(data);
  };
}

TFE_MonitoringStringGauge0* TFE_MonitoringNewStringGauge0(
    const char* name, TF_Status* status, const char* description) {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_29_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_29(mht_29_v, 529, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewStringGauge0");

  auto* result = new TFE_MonitoringStringGauge0({name, description});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge0(TFE_MonitoringStringGauge0* gauge) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_30(mht_30_v, 542, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteStringGauge0");

  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge0(
    TFE_MonitoringStringGauge0* gauge) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_31(mht_31_v, 550, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellStringGauge0");

  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell()));
}

TFE_MonitoringStringGauge1* TFE_MonitoringNewStringGauge1(
    const char* name, TF_Status* status, const char* description,
    const char* label1) {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_32_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_32_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_32(mht_32_v, 563, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewStringGauge1");

  auto* result = new TFE_MonitoringStringGauge1({name, description, label1});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge1(TFE_MonitoringStringGauge1* gauge) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_33(mht_33_v, 576, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteStringGauge1");

  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge1(
    TFE_MonitoringStringGauge1* gauge, const char* label1) {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_34(mht_34_v, 585, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellStringGauge1");

  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1)));
}

TFE_MonitoringStringGauge2* TFE_MonitoringNewStringGauge2(
    const char* name, TF_Status* status, const char* description,
    const char* label1, const char* label2) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_35_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_35_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_35_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_35(mht_35_v, 599, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewStringGauge2");

  auto* result =
      new TFE_MonitoringStringGauge2({name, description, label1, label2});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge2(TFE_MonitoringStringGauge2* gauge) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_36(mht_36_v, 613, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteStringGauge2");

  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge2(
    TFE_MonitoringStringGauge2* gauge, const char* label1, const char* label2) {
   std::vector<std::string> mht_37_v;
   mht_37_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_37_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_37(mht_37_v, 623, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellStringGauge2");

  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2)));
}

TFE_MonitoringStringGauge3* TFE_MonitoringNewStringGauge3(
    const char* name, TF_Status* status, const char* description,
    const char* label1, const char* label2, const char* label3) {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_38_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_38_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_38_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   mht_38_v.push_back("label3: \"" + (label3 == nullptr ? std::string("nullptr") : std::string((char*)label3)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_38(mht_38_v, 638, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewStringGauge3");

  auto* result = new TFE_MonitoringStringGauge3(
      {name, description, label1, label2, label3});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge3(TFE_MonitoringStringGauge3* gauge) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_39(mht_39_v, 652, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteStringGauge3");

  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge3(
    TFE_MonitoringStringGauge3* gauge, const char* label1, const char* label2,
    const char* label3) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_40_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   mht_40_v.push_back("label3: \"" + (label3 == nullptr ? std::string("nullptr") : std::string((char*)label3)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_40(mht_40_v, 664, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellStringGauge3");

  return static_cast<TFE_MonitoringStringGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2, label3)));
}

TFE_MonitoringStringGauge4* TFE_MonitoringNewStringGauge4(
    const char* name, TF_Status* status, const char* description,
    const char* label1, const char* label2, const char* label3,
    const char* label4) {
   std::vector<std::string> mht_41_v;
   mht_41_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_41_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_41_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_41_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   mht_41_v.push_back("label3: \"" + (label3 == nullptr ? std::string("nullptr") : std::string((char*)label3)) + "\"");
   mht_41_v.push_back("label4: \"" + (label4 == nullptr ? std::string("nullptr") : std::string((char*)label4)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_41(mht_41_v, 681, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewStringGauge4");

  auto* result = new TFE_MonitoringStringGauge4(
      {name, description, label1, label2, label3, label4});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteStringGauge4(TFE_MonitoringStringGauge4* gauge) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_42(mht_42_v, 695, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteStringGauge4");

  delete gauge;
}

TFE_MonitoringStringGaugeCell* TFE_MonitoringGetCellStringGauge4(
    TFE_MonitoringStringGauge4* gauge, const char* label1, const char* label2,
    const char* label3, const char* label4) {
   std::vector<std::string> mht_43_v;
   mht_43_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_43_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   mht_43_v.push_back("label3: \"" + (label3 == nullptr ? std::string("nullptr") : std::string((char*)label3)) + "\"");
   mht_43_v.push_back("label4: \"" + (label4 == nullptr ? std::string("nullptr") : std::string((char*)label4)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_43(mht_43_v, 708, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellStringGauge4");

  return static_cast<TFE_MonitoringStringGaugeCell*>(static_cast<void*>(
      gauge->gauge->GetCell(label1, label2, label3, label4)));
}

void TFE_MonitoringBoolGaugeCellSet(TFE_MonitoringBoolGaugeCell* cell,
                                    bool value) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_44(mht_44_v, 717, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringBoolGaugeCellSet");

  cell->cell.Set(value);
}

bool TFE_MonitoringBoolGaugeCellValue(TFE_MonitoringBoolGaugeCell* cell) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_45(mht_45_v, 724, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringBoolGaugeCellValue");

  return cell->cell.value();
}

TFE_MonitoringBoolGauge0* TFE_MonitoringNewBoolGauge0(const char* name,
                                                      TF_Status* status,
                                                      const char* description) {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_46_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_46(mht_46_v, 735, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewBoolGauge0");

  auto* result = new TFE_MonitoringBoolGauge0({name, description});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteBoolGauge0(TFE_MonitoringBoolGauge0* gauge) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_47(mht_47_v, 748, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteBoolGauge0");

  delete gauge;
}

TFE_MonitoringBoolGaugeCell* TFE_MonitoringGetCellBoolGauge0(
    TFE_MonitoringBoolGauge0* gauge) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_48(mht_48_v, 756, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellBoolGauge0");

  return static_cast<TFE_MonitoringBoolGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell()));
}

TFE_MonitoringBoolGauge1* TFE_MonitoringNewBoolGauge1(const char* name,
                                                      TF_Status* status,
                                                      const char* description,
                                                      const char* label1) {
   std::vector<std::string> mht_49_v;
   mht_49_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_49_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_49_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_49(mht_49_v, 770, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewBoolGauge1");

  auto* result = new TFE_MonitoringBoolGauge1({name, description, label1});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteBoolGauge1(TFE_MonitoringBoolGauge1* gauge) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_50(mht_50_v, 783, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteBoolGauge1");

  delete gauge;
}

TFE_MonitoringBoolGaugeCell* TFE_MonitoringGetCellBoolGauge1(
    TFE_MonitoringBoolGauge1* gauge, const char* label1) {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_51(mht_51_v, 792, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellBoolGauge1");

  return static_cast<TFE_MonitoringBoolGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1)));
}

TFE_MonitoringBoolGauge2* TFE_MonitoringNewBoolGauge2(const char* name,
                                                      TF_Status* status,
                                                      const char* description,
                                                      const char* label1,
                                                      const char* label2) {
   std::vector<std::string> mht_52_v;
   mht_52_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_52_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_52_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_52_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_52(mht_52_v, 808, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewBoolGauge2");

  auto* result =
      new TFE_MonitoringBoolGauge2({name, description, label1, label2});
  Set_TF_Status_from_Status(status, result->gauge->GetStatus());
  if (!result->gauge->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteBoolGauge2(TFE_MonitoringBoolGauge2* gauge) {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_53(mht_53_v, 822, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteBoolGauge2");

  delete gauge;
}

TFE_MonitoringBoolGaugeCell* TFE_MonitoringGetCellBoolGauge2(
    TFE_MonitoringBoolGauge2* gauge, const char* label1, const char* label2) {
   std::vector<std::string> mht_54_v;
   mht_54_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_54_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_54(mht_54_v, 832, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellBoolGauge2");

  return static_cast<TFE_MonitoringBoolGaugeCell*>(
      static_cast<void*>(gauge->gauge->GetCell(label1, label2)));
}

void TFE_MonitoringSamplerCellAdd(TFE_MonitoringSamplerCell* cell,
                                  double value) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_55(mht_55_v, 841, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringSamplerCellAdd");

  cell->cell.Add(value);
}

void TFE_MonitoringSamplerCellValue(TFE_MonitoringSamplerCell* cell,
                                    TF_Buffer* buf) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_56(mht_56_v, 849, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringSamplerCellValue");

  string content;
  cell->cell.value().SerializeToString(&content);
  void* data = tensorflow::port::Malloc(content.length());
  content.copy(static_cast<char*>(data), content.length(), 0);
  buf->data = data;
  buf->length = content.length();
  buf->data_deallocator = [](void* data, size_t length) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_57(mht_57_v, 859, "", "./tensorflow/c/eager/c_api_experimental.cc", "lambda");

    tensorflow::port::Free(data);
  };
}

TFE_MonitoringBuckets* TFE_MonitoringNewExponentialBuckets(double scale,
                                                           double growth_factor,
                                                           int bucket_count) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_58(mht_58_v, 869, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewExponentialBuckets");

  return new TFE_MonitoringBuckets([scale, growth_factor, bucket_count]() {
    return tensorflow::monitoring::Buckets::Exponential(scale, growth_factor,
                                                        bucket_count);
  });
}

void TFE_MonitoringDeleteBuckets(TFE_MonitoringBuckets* buckets) {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_59(mht_59_v, 879, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteBuckets");

  delete buckets;
}

TFE_MonitoringSampler0* TFE_MonitoringNewSampler0(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* status,
    const char* description) {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_60_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_60(mht_60_v, 890, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewSampler0");

  auto* result = new TFE_MonitoringSampler0(
      {name, buckets->create_buckets(), description});
  Set_TF_Status_from_Status(status, result->sampler->GetStatus());
  if (!result->sampler->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteSampler0(TFE_MonitoringSampler0* sampler) {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_61(mht_61_v, 904, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteSampler0");

  delete sampler;
}

TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler0(
    TFE_MonitoringSampler0* sampler) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_62(mht_62_v, 912, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellSampler0");

  return static_cast<TFE_MonitoringSamplerCell*>(
      static_cast<void*>(sampler->sampler->GetCell()));
}

TFE_MonitoringSampler1* TFE_MonitoringNewSampler1(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* status,
    const char* description, const char* label1) {
   std::vector<std::string> mht_63_v;
   mht_63_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_63_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_63_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_63(mht_63_v, 925, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewSampler1");

  auto* result = new TFE_MonitoringSampler1(
      {name, buckets->create_buckets(), description, label1});
  Set_TF_Status_from_Status(status, result->sampler->GetStatus());
  if (!result->sampler->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteSampler1(TFE_MonitoringSampler1* sampler) {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_64(mht_64_v, 939, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteSampler1");

  delete sampler;
}

TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler1(
    TFE_MonitoringSampler1* sampler, const char* label1) {
   std::vector<std::string> mht_65_v;
   mht_65_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_65(mht_65_v, 948, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellSampler1");

  return static_cast<TFE_MonitoringSamplerCell*>(
      static_cast<void*>(sampler->sampler->GetCell(label1)));
}

TFE_MonitoringSampler2* TFE_MonitoringNewSampler2(
    const char* name, TFE_MonitoringBuckets* buckets, TF_Status* status,
    const char* description, const char* label1, const char* label2) {
   std::vector<std::string> mht_66_v;
   mht_66_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   mht_66_v.push_back("description: \"" + (description == nullptr ? std::string("nullptr") : std::string((char*)description)) + "\"");
   mht_66_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_66_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_66(mht_66_v, 962, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringNewSampler2");

  auto* result = new TFE_MonitoringSampler2(
      {name, buckets->create_buckets(), description, label1, label2});
  Set_TF_Status_from_Status(status, result->sampler->GetStatus());
  if (!result->sampler->GetStatus().ok()) {
    delete result;
    return nullptr;
  }
  return result;
}

void TFE_MonitoringDeleteSampler2(TFE_MonitoringSampler2* sampler) {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_67(mht_67_v, 976, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringDeleteSampler2");

  delete sampler;
}

TFE_MonitoringSamplerCell* TFE_MonitoringGetCellSampler2(
    TFE_MonitoringSampler2* sampler, const char* label1, const char* label2) {
   std::vector<std::string> mht_68_v;
   mht_68_v.push_back("label1: \"" + (label1 == nullptr ? std::string("nullptr") : std::string((char*)label1)) + "\"");
   mht_68_v.push_back("label2: \"" + (label2 == nullptr ? std::string("nullptr") : std::string((char*)label2)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_68(mht_68_v, 986, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_MonitoringGetCellSampler2");

  return static_cast<TFE_MonitoringSamplerCell*>(
      static_cast<void*>(sampler->sampler->GetCell(label1, label2)));
}

void TFE_ContextOptionsSetTfrt(TFE_ContextOptions* options, bool use_tfrt) {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_69(mht_69_v, 994, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextOptionsSetTfrt");

  options->use_tfrt = use_tfrt;
}

void TFE_ContextOptionsSetTfrtDistributedRuntime(
    TFE_ContextOptions* options, bool use_tfrt_distributed_runtime) {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_70(mht_70_v, 1002, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextOptionsSetTfrtDistributedRuntime");

  options->use_tfrt_distributed_runtime = use_tfrt_distributed_runtime;
}

TFE_CancellationManager* TFE_NewCancellationManager() {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_71(mht_71_v, 1009, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_NewCancellationManager");

  return tensorflow::wrap(new tensorflow::CancellationManager);
}

void TFE_CancellationManagerStartCancel(
    TFE_CancellationManager* cancellation_manager) {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_72(mht_72_v, 1017, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_CancellationManagerStartCancel");

  tensorflow::unwrap(cancellation_manager)->StartCancel();
}

bool TFE_CancellationManagerIsCancelled(
    TFE_CancellationManager* cancellation_manager) {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_73(mht_73_v, 1025, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_CancellationManagerIsCancelled");

  return tensorflow::unwrap(cancellation_manager)->IsCancelled();
}

void TFE_DeleteCancellationManager(
    TFE_CancellationManager* cancellation_manager) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_74(mht_74_v, 1033, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_DeleteCancellationManager");

  delete tensorflow::unwrap(cancellation_manager);
}

void TFE_OpSetCancellationManager(TFE_Op* op,
                                  TFE_CancellationManager* cancellation_manager,
                                  TF_Status* status) {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_75(mht_75_v, 1042, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_OpSetCancellationManager");

  tensorflow::unwrap(op)->SetCancellationManager(
      tensorflow::unwrap(cancellation_manager));
  status->status = tensorflow::Status::OK();
}

TFE_Executor* TFE_NewExecutor(bool is_async) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_76(mht_76_v, 1051, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_NewExecutor");

  return new TFE_Executor(is_async);
}

void TFE_DeleteExecutor(TFE_Executor* executor) {
   std::vector<std::string> mht_77_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_77(mht_77_v, 1058, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_DeleteExecutor");
 delete executor; }

bool TFE_ExecutorIsAsync(TFE_Executor* executor) {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_78(mht_78_v, 1063, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ExecutorIsAsync");

  return executor->executor()->Async();
}

void TFE_ExecutorWaitForAllPendingNodes(TFE_Executor* executor,
                                        TF_Status* status) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_79(mht_79_v, 1071, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ExecutorWaitForAllPendingNodes");

  status->status = executor->executor()->WaitForAllPendingNodes();
}

void TFE_ExecutorClearError(TFE_Executor* executor) {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_80(mht_80_v, 1078, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ExecutorClearError");

  executor->executor()->ClearError();
}

void TFE_ContextSetExecutorForThread(TFE_Context* ctx, TFE_Executor* executor) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_81(mht_81_v, 1085, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextSetExecutorForThread");

  tensorflow::unwrap(ctx)->SetExecutorForThread(executor->executor());
}

TFE_Executor* TFE_ContextGetExecutorForThread(TFE_Context* ctx) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_82(mht_82_v, 1092, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextGetExecutorForThread");

  return new TFE_Executor(&tensorflow::unwrap(ctx)->Executor());
}

void TFE_HostAddressSpace(TFE_Context* ctx, TF_Buffer* buf) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_83(mht_83_v, 1099, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_HostAddressSpace");

  auto address_space = tensorflow::DeviceNameUtils::AddressSpace(
      tensorflow::unwrap(ctx)->HostCPUParsedName());
  auto str = tensorflow::DeviceNameUtils::ParsedNameToString(address_space);
  void* data = tensorflow::port::Malloc(str.length());
  str.copy(static_cast<char*>(data), str.length(), 0);
  buf->data = data;
  buf->length = str.length();
  buf->data_deallocator = [](void* data, size_t length) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_84(mht_84_v, 1110, "", "./tensorflow/c/eager/c_api_experimental.cc", "lambda");

    tensorflow::port::Free(data);
  };
}

void TFE_ContextGetFunctionDef(TFE_Context* ctx, const char* function_name,
                               TF_Buffer* buf, TF_Status* status) {
   std::vector<std::string> mht_85_v;
   mht_85_v.push_back("function_name: \"" + (function_name == nullptr ? std::string("nullptr") : std::string((char*)function_name)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_85(mht_85_v, 1120, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextGetFunctionDef");

  auto* function_def = tensorflow::unwrap(ctx)->FindFunctionDef(function_name);
  if (function_def == nullptr) {
    status->status = tensorflow::errors::NotFound(
        "Unable to find FunctionDef with name: ", function_name);
    return;
  }
  string str = function_def->SerializeAsString();
  void* data = tensorflow::port::Malloc(str.length());
  str.copy(static_cast<char*>(data), str.length(), 0);
  buf->data = data;
  buf->length = str.length();
  buf->data_deallocator = [](void* data, size_t length) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_86(mht_86_v, 1135, "", "./tensorflow/c/eager/c_api_experimental.cc", "lambda");

    tensorflow::port::Free(data);
  };
  status->status = tensorflow::Status::OK();
}

TF_Tensor* TFE_AllocateHostTensor(TFE_Context* ctx, TF_DataType dtype,
                                  const int64_t* dims, int num_dims,
                                  TF_Status* status) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_87(mht_87_v, 1146, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_AllocateHostTensor");

  std::vector<int64_t> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<int64_t>(dims[i]);
  }

  if (ctx == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid Context");
    return nullptr;
  }

  tensorflow::AbstractTensorInterface* t =
      tensorflow::unwrap(ctx)->CreateTensor(
          static_cast<tensorflow::DataType>(dtype), dimvec);

  if (t == nullptr) {
    status->status =
        tensorflow::errors::InvalidArgument("Unsupported dtype: ", dtype);
    return nullptr;
  }

  return new TF_Tensor{t};
}

TFE_TensorHandle* TFE_NewTensorHandleFromTensor(TFE_Context* ctx, TF_Tensor* t,
                                                TF_Status* status) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_88(mht_88_v, 1174, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_NewTensorHandleFromTensor");

  return tensorflow::wrap(
      tensorflow::unwrap(ctx)->CreateLocalHandle(t->tensor));
}

TFE_TensorHandle* TFE_CreatePackedTensorHandle(TFE_Context* ctx,
                                               TFE_TensorHandle** handles,
                                               int* num_handles,
                                               TF_Status* status) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_89(mht_89_v, 1185, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_CreatePackedTensorHandle");

  std::vector<tensorflow::TensorHandle*> tensor_handles;
  tensor_handles.reserve(*num_handles);
  for (int i = 0; i < *num_handles; ++i) {
    tensorflow::ImmediateExecutionTensorHandle* unwrapped_handle =
        tensorflow::unwrap(handles[i]);
    if (tensorflow::CustomDeviceTensorHandle::classof(unwrapped_handle)) {
      // One of the inputs we're trying to pack is on a custom device. We'll let
      // the first custom device we see handle all of the packing.
      auto* custom_device_handle =
          tensorflow::down_cast<tensorflow::CustomDeviceTensorHandle*>(
              unwrapped_handle);
      tensorflow::ImmediateExecutionTensorHandle* result;
      status->status = custom_device_handle->device()->Pack(
          absl::Span<tensorflow::ImmediateExecutionTensorHandle*>(
              tensorflow::unwrap(handles), *num_handles),
          &result);
      return tensorflow::wrap(result);
    }
    tensor_handles.push_back(
        tensorflow::TensorHandleFromInterface(unwrapped_handle));
  }
  tensorflow::EagerContext* context =
      tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
  tensorflow::TensorHandle* handle = nullptr;
  status->status = tensorflow::TensorHandle::CreatePackedHandle(
      std::move(tensor_handles), context, &handle);
  return tensorflow::wrap(handle);
}

void TFE_ContextSetSoftDevicePlacement(TFE_Context* ctx, unsigned char enable,
                                       TF_Status* status) {
   std::vector<std::string> mht_90_v;
   mht_90_v.push_back("enable: '" + std::string(1, enable) + "'");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_90(mht_90_v, 1220, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextSetSoftDevicePlacement");

  tensorflow::unwrap(ctx)->SetAllowSoftPlacement(enable);
}

void TFE_ContextSetLogDevicePlacement(TFE_Context* ctx, unsigned char enable,
                                      TF_Status* status) {
   std::vector<std::string> mht_91_v;
   mht_91_v.push_back("enable: '" + std::string(1, enable) + "'");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_91(mht_91_v, 1229, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextSetLogDevicePlacement");

  tensorflow::unwrap(ctx)->SetLogDevicePlacement(enable);
}

void TFE_ContextSetRunEagerOpAsFunction(TFE_Context* ctx, unsigned char enable,
                                        TF_Status* status) {
   std::vector<std::string> mht_92_v;
   mht_92_v.push_back("enable: '" + std::string(1, enable) + "'");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_92(mht_92_v, 1238, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextSetRunEagerOpAsFunction");

  tensorflow::unwrap(ctx)->SetRunEagerOpAsFunction(enable);
}

void TFE_ContextSetJitCompileRewrite(TFE_Context* ctx, unsigned char enable,
                                     TF_Status* status) {
   std::vector<std::string> mht_93_v;
   mht_93_v.push_back("enable: '" + std::string(1, enable) + "'");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_93(mht_93_v, 1247, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ContextSetJitCompileRewrite");

  tensorflow::unwrap(ctx)->SetJitCompileRewrite(enable);
}

const char* TFE_TensorHandleDeviceType(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_94(mht_94_v, 1254, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_TensorHandleDeviceType");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return nullptr;
  }
  return tensorflow::unwrap(h)->DeviceType(&status->status);
}

int TFE_TensorHandleDeviceID(TFE_TensorHandle* h, TF_Status* status) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_95(mht_95_v, 1265, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_TensorHandleDeviceID");

  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument("Invalid handle");
    return -1;
  }
  return tensorflow::unwrap(h)->DeviceId(&status->status);
}

TF_CAPI_EXPORT extern void TFE_TensorHandleGetStatus(TFE_TensorHandle* h,
                                                     TF_Status* status) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_96(mht_96_v, 1277, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_TensorHandleGetStatus");

  status->status = tensorflow::unwrap(h)->TensorHandleStatus();
}

void TFE_GetExecutedOpNames(TFE_Context* ctx, TF_Buffer* buf,
                            TF_Status* status) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_97(mht_97_v, 1285, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_GetExecutedOpNames");

  const std::vector<std::string>& op_names =
      tensorflow::unwrap(ctx)->GetLoggedOpsTestonly();

  std::ostringstream op_names_oss;
  for (const auto& op : op_names) {
    op_names_oss << op << ", ";
  }
  const std::string& op_names_str = op_names_oss.str();
  void* data = tensorflow::port::Malloc(op_names_str.length());
  op_names_str.copy(static_cast<char*>(data), op_names_str.length(), 0);
  buf->data = data;
  buf->length = op_names_str.length();
  buf->data_deallocator = [](void* data, size_t length) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_98(mht_98_v, 1301, "", "./tensorflow/c/eager/c_api_experimental.cc", "lambda");

    tensorflow::port::Free(data);
  };
  status->status = tensorflow::Status::OK();
}

void TFE_SetLogicalCpuDevices(TFE_Context* ctx, int num_cpus,
                              const char* prefix, TF_Status* status) {
   std::vector<std::string> mht_99_v;
   mht_99_v.push_back("prefix: \"" + (prefix == nullptr ? std::string("nullptr") : std::string((char*)prefix)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_99(mht_99_v, 1312, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_SetLogicalCpuDevices");

  std::vector<std::unique_ptr<tensorflow::Device>> devices;

  if (prefix == nullptr || strlen(prefix) == 0)
    prefix = "/job:localhost/replica:0/task:0";

  tensorflow::SessionOptions sess_options;
  (*sess_options.config.mutable_device_count())["CPU"] = num_cpus;
  status->status =
      tensorflow::DeviceFactory::AddCpuDevices(sess_options, prefix, &devices);

  // Remove the device that has the host device name since host device is alreay
  // in an initialized context.
  for (auto d = devices.begin(); d != devices.end();) {
    if (absl::StrContains(d->get()->name(), "CPU:0")) {
      d = devices.erase(d);
    } else {
      ++d;
    }
  }

  status->status = tensorflow::unwrap(ctx)->AddDevices(std::move(devices));
}

void TFE_InsertConfigKeyValue(TFE_Context* ctx, const char* key,
                              const char* value, TF_Status* status) {
   std::vector<std::string> mht_100_v;
   mht_100_v.push_back("key: \"" + (key == nullptr ? std::string("nullptr") : std::string((char*)key)) + "\"");
   mht_100_v.push_back("value: \"" + (value == nullptr ? std::string("nullptr") : std::string((char*)value)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_100(mht_100_v, 1342, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_InsertConfigKeyValue");

  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tensorflow::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service agent is not enabled.");
    return;
  }
  status->status = coord_agent->InsertKeyValue(key, value);
}

void TFE_GetConfigKeyValue(TFE_Context* ctx, const char* key,
                           TF_Buffer* value_buf, TF_Status* status) {
   std::vector<std::string> mht_101_v;
   mht_101_v.push_back("key: \"" + (key == nullptr ? std::string("nullptr") : std::string((char*)key)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_101(mht_101_v, 1360, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_GetConfigKeyValue");

  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tensorflow::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service is not enabled.");
    return;
  }
  auto status_or_value = coord_agent->GetKeyValue(key);
  status->status = status_or_value.status();
  if (!status_or_value.ok()) return;

  const std::string& value_string = status_or_value.ValueOrDie();
  void* data = tensorflow::port::Malloc(value_string.length());
  value_string.copy(static_cast<char*>(data), value_string.length(), 0);
  value_buf->data = data;
  value_buf->length = value_string.length();
  value_buf->data_deallocator = [](void* data, size_t length) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_102(mht_102_v, 1382, "", "./tensorflow/c/eager/c_api_experimental.cc", "lambda");

    tensorflow::port::Free(data);
  };
}

void TFE_DeleteConfigKeyValue(TFE_Context* ctx, const char* key,
                              TF_Status* status) {
   std::vector<std::string> mht_103_v;
   mht_103_v.push_back("key: \"" + (key == nullptr ? std::string("nullptr") : std::string((char*)key)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_103(mht_103_v, 1392, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_DeleteConfigKeyValue");

  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tensorflow::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service is not enabled.");
    return;
  }
  status->status = coord_agent->DeleteKeyValue(key);
}

void TFE_ReportErrorToCluster(TFE_Context* ctx, int error_code,
                              const char* error_message, TF_Status* status) {
   std::vector<std::string> mht_104_v;
   mht_104_v.push_back("error_message: \"" + (error_message == nullptr ? std::string("nullptr") : std::string((char*)error_message)) + "\"");
   MHTracer_DTPStensorflowPScPSeagerPSc_api_experimentalDTcc mht_104(mht_104_v, 1410, "", "./tensorflow/c/eager/c_api_experimental.cc", "TFE_ReportErrorToCluster");

  tensorflow::ImmediateExecutionDistributedManager* dist_mgr =
      tensorflow::unwrap(ctx)->GetDistributedManager();
  tensorflow::CoordinationServiceAgent* coord_agent =
      dist_mgr->GetCoordinationServiceAgent();
  if (coord_agent == nullptr) {
    status->status = tensorflow::errors::FailedPrecondition(
        "Coordination service is not enabled.");
    return;
  }
  tensorflow::Status s(static_cast<tensorflow::error::Code>(error_code),
                       error_message);
  status->status = coord_agent->ReportError(s);
}
