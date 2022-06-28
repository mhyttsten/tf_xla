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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPStypes_utilDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPStypes_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPStypes_utilDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/metrics/types_util.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace {

// Extracts information from mlir::FileLineColLoc to the proto message
// tflite::metrics::ConverterErrorData::FileLoc.
void ExtractFileLine(const FileLineColLoc& loc,
                     tflite::metrics::ConverterErrorData::FileLoc* fileline) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPStypes_utilDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/mlir/lite/metrics/types_util.cc", "ExtractFileLine");

  fileline->set_filename(loc.getFilename().str());
  fileline->set_line(loc.getLine());
  fileline->set_column(loc.getColumn());
}

// Defines a child class of Location to access its protected members.
class LocationExtractor : public Location {
 public:
  explicit LocationExtractor(const Location& loc) : Location(loc) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPStypes_utilDTcc mht_1(mht_1_v, 208, "", "./tensorflow/compiler/mlir/lite/metrics/types_util.cc", "LocationExtractor");
}

  void Extract(tflite::metrics::ConverterErrorData* error_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPStypes_utilDTcc mht_2(mht_2_v, 213, "", "./tensorflow/compiler/mlir/lite/metrics/types_util.cc", "Extract");

    using tflite::metrics::ConverterErrorData;
    auto mutable_location = error_data->mutable_location();

    llvm::TypeSwitch<LocationAttr>(impl)
        .Case<OpaqueLoc>([&](OpaqueLoc loc) {
          LocationExtractor(loc.getFallbackLocation()).Extract(error_data);
        })
        .Case<UnknownLoc>([&](UnknownLoc loc) {
          mutable_location->set_type(ConverterErrorData::UNKNOWNLOC);
        })
        .Case<FileLineColLoc>([&](FileLineColLoc loc) {
          if (!mutable_location->has_type()) {
            mutable_location->set_type(ConverterErrorData::CALLSITELOC);
          }
          auto new_call = mutable_location->mutable_call()->Add();
          ExtractFileLine(loc, new_call->mutable_source());
        })
        .Case<NameLoc>([&](NameLoc loc) {
          if (!mutable_location->has_type()) {
            mutable_location->set_type(ConverterErrorData::NAMELOC);
          }

          auto new_call = mutable_location->mutable_call()->Add();
          new_call->set_name(loc.getName().str());
          // Add child as the source location.
          auto child_loc = loc.getChildLoc();
          if (child_loc.isa<FileLineColLoc>()) {
            auto typed_child_loc = child_loc.dyn_cast<FileLineColLoc>();
            ExtractFileLine(typed_child_loc, new_call->mutable_source());
          }
        })
        .Case<CallSiteLoc>([&](CallSiteLoc loc) {
          mutable_location->set_type(ConverterErrorData::CALLSITELOC);
          LocationExtractor(loc.getCallee()).Extract(error_data);
          LocationExtractor(loc.getCaller()).Extract(error_data);
        })
        .Case<FusedLoc>([&](FusedLoc loc) {
          auto locations = loc.getLocations();
          size_t num_locs = locations.size();
          // Skip the first location if it stores information for propagating
          // op_type metadata.
          if (num_locs > 0) {
            if (auto name_loc = locations[0].dyn_cast<mlir::NameLoc>()) {
              if (name_loc.getName().strref().endswith(":")) {
                if (num_locs == 2) {
                  return LocationExtractor(locations[1]).Extract(error_data);
                } else if (num_locs > 2) {
                  locations = {locations.begin() + 1, locations.end()};
                }
              }
            }
          }

          mutable_location->set_type(ConverterErrorData::FUSEDLOC);
          llvm::interleave(
              locations,
              [&](Location l) { LocationExtractor(l).Extract(error_data); },
              [&]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPStypes_utilDTcc mht_3(mht_3_v, 274, "", "./tensorflow/compiler/mlir/lite/metrics/types_util.cc", "lambda");
});
        });
  }
};
}  // namespace

tflite::metrics::ConverterErrorData NewConverterErrorData(
    const std ::string& pass_name, const std::string& error_message,
    tflite::metrics::ConverterErrorData::ErrorCode error_code,
    const std::string& op_name, const Location& location) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("error_message: \"" + error_message + "\"");
   mht_4_v.push_back("op_name: \"" + op_name + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSmetricsPStypes_utilDTcc mht_4(mht_4_v, 288, "", "./tensorflow/compiler/mlir/lite/metrics/types_util.cc", "NewConverterErrorData");

  using tflite::metrics::ConverterErrorData;
  ConverterErrorData error;
  if (!pass_name.empty()) {
    error.set_subcomponent(pass_name);
  }

  if (!error_message.empty()) {
    error.set_error_message(error_message);
  }

  if (!op_name.empty()) {
    error.mutable_operator_()->set_name(op_name);
  }

  error.set_error_code(error_code);
  LocationExtractor(location).Extract(&error);
  return error;
}

}  // namespace TFL
}  // namespace mlir
