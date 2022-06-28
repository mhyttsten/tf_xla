/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_DEBUG_OPTIONS_FLAGS_H_
#define TENSORFLOW_COMPILER_XLA_DEBUG_OPTIONS_FLAGS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSdebug_options_flagsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSdebug_options_flagsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSdebug_options_flagsDTh() {
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


#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {

// Appends flag definitions for debug options to flag_list.
void AppendDebugOptionsFlags(std::vector<tensorflow::Flag>* flag_list);

// Fetches a DebugOptions proto message from flags provided to the program.
// Flags must be registered with the flags parser using AppendDebugOptionsFlags
// first.
DebugOptions GetDebugOptionsFromFlags();

// Gets a DebugOptions proto that reflects the defaults as if no flags were set.
DebugOptions DefaultDebugOptionsIgnoringFlags();

// Consumes a unit of "compiler fuel" for the given pass, and returns false if
// we're out of fuel for that pass.
//
// Compiler fuel is a debugging tool useful for bisecting compiler passes.  Each
// time a pass "does something", it consumes a unit of fuel, and once it's out
// of fuel, it stops doing any transformations.  This way if you suspect a pass
// has a bug, you can bisect the amount of fuel it gets and find exactly which
// change causes the problem.
//
// The very first time a pass runs out of fuel, `just_ran_out` is set to true.
// This lets you take action (e.g. log a message).  But see also the convenience
// overload below.
//
// By default all passes have infinite fuel.  You can restrict how much fuel a
// pass has by specifying XLA_FLAGS=--xla_fuel=PASS1=NUM1,PASS2=NUM2,...
//
// If a user specifies --xla_fuel=PASS=NUM but ConsumeFuel(PASS) is not called
// before the program exits, we'll print a warning.
//
// We recommend as a convention you use a pass's name for the `pass` argument,
// but any value is accepted.
bool ConsumeFuel(absl::string_view pass, bool* just_ran_out = nullptr);

// Overload of ConsumeFuel that lets you pass in a functor which generates a log
// message when we first run out of fuel for a pass.  This is useful because
// you're usually interested in what *would have* happened right when we ran out
// of fuel.
//
// Example usage:
//
//   if (ConsumeFuel("pass-name", [&] { return "Not fooing bar."; })) {
//     return;
//   }
//
template <typename MsgGenerator>
bool ConsumeFuel(absl::string_view pass,
                 const MsgGenerator& ran_out_of_fuel_msg) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("pass: \"" + std::string(pass.data(), pass.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSdebug_options_flagsDTh mht_0(mht_0_v, 245, "", "./tensorflow/compiler/xla/debug_options_flags.h", "ConsumeFuel");

  bool just_ran_out = false;
  bool ret = ConsumeFuel(pass, &just_ran_out);
  if (just_ran_out) {
    LOG(ERROR) << "Out of fuel for \"" << pass
               << "\": " << ran_out_of_fuel_msg();
  }
  return ret;
}

// By default compiler fuel is global; if you run two compiler threads, they
// will consume from the same fuel pool.
//
// Calling this function changes the behavior of fuel for the current thread:
// From this point onward, it will use a private fuel pool.  The thread-local
// fuel pool is initialized to the values the global fuel pool had at process
// startup.
//
// You may call this function twice in the same thread to reset its fuel pool
// back to the initial state.
void ResetThreadLocalFuel();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_DEBUG_OPTIONS_FLAGS_H_
