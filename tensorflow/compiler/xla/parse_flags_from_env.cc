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
class MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc() {
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

// This module exports ParseFlagsFromEnvAndDieIfUnknown(), which allows other
// modules to parse flags from an environtment variable, or a file named by the
// environment variable.

#include "tensorflow/compiler/xla/parse_flags_from_env.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {

static const char kWS[] = " \t\r\n";  // whitespace

// The following struct represents an argv[]-style array, parsed
// from data gleaned from the environment.
//
// As usual, an anonymous namespace is advisable to avoid
// constructor/destructor collisions with other "private" types
// in the same named namespace.
namespace {

// Functor which deletes objects by calling `free`.  Necessary to free strdup'ed
// strings created by AppendToEnvArgv.
struct FreeDeleter {
  void operator()(char* ptr) { free(ptr); }
};

struct EnvArgv {
  EnvArgv() : initialized(false), argc(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc mht_0(mht_0_v, 227, "", "./tensorflow/compiler/xla/parse_flags_from_env.cc", "EnvArgv");
}
  bool initialized;         // whether the other fields have been set.
  int argc;                 // elements used in argv[]
  std::vector<char*> argv;  // flag arguments parsed from environment string.
  // saved values from argv[] to avoid leaks
  std::vector<std::unique_ptr<char, FreeDeleter>> argv_save;
};
}  // anonymous namespace

// Append the string s0[0, .., s0len-1] concatenated with s1[0, .., s1len-1] as
// a newly allocated nul-terminated string to the array *a.  If s0==nullptr, a
// nullptr is appended without increasing a->argc.
static void AppendToEnvArgv(const char* s0, size_t s0len, const char* s1,
                            size_t s1len, EnvArgv* a) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("s0: \"" + (s0 == nullptr ? std::string("nullptr") : std::string((char*)s0)) + "\"");
   mht_1_v.push_back("s1: \"" + (s1 == nullptr ? std::string("nullptr") : std::string((char*)s1)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc mht_1(mht_1_v, 245, "", "./tensorflow/compiler/xla/parse_flags_from_env.cc", "AppendToEnvArgv");

  if (s0 == nullptr) {
    a->argv.push_back(nullptr);
    a->argv_save.push_back(nullptr);
  } else {
    std::string s = std::string(s0, s0len) + std::string(s1, s1len);
    char* str = strdup(s.c_str());
    a->argv.push_back(str);
    a->argv_save.emplace_back(str);
    a->argc++;
  }
}

// Like s.find_first_of(x, pos), but return s.size() when find_first_of() would
// return std::string::npos.  This avoids if-statements elsewhere.
static size_t FindFirstOf(const std::string& s, const char* x, size_t pos) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("s: \"" + s + "\"");
   mht_2_v.push_back("x: \"" + (x == nullptr ? std::string("nullptr") : std::string((char*)x)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc mht_2(mht_2_v, 265, "", "./tensorflow/compiler/xla/parse_flags_from_env.cc", "FindFirstOf");

  size_t result = s.find_first_of(x, pos);
  return result == std::string::npos ? s.size() : result;
}

// Like s.find_first_not_of(x, pos), but return s.size() when
// find_first_not_of() would return std::string::npos.  This avoids
// if-statements elsewhere.
static size_t FindFirstNotOf(const std::string& s, const char* x, size_t pos) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("s: \"" + s + "\"");
   mht_3_v.push_back("x: \"" + (x == nullptr ? std::string("nullptr") : std::string((char*)x)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc mht_3(mht_3_v, 278, "", "./tensorflow/compiler/xla/parse_flags_from_env.cc", "FindFirstNotOf");

  size_t result = s.find_first_not_of(x, pos);
  return result == std::string::npos ? s.size() : result;
}

// Given a string containing flags, parse them into the XLA command line flags.
// The parse is best effort, and gives up on the first syntax error.
static void ParseArgvFromString(const std::string& flag_str, EnvArgv* a) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("flag_str: \"" + flag_str + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc mht_4(mht_4_v, 289, "", "./tensorflow/compiler/xla/parse_flags_from_env.cc", "ParseArgvFromString");

  size_t b = FindFirstNotOf(flag_str, kWS, 0);
  while (b != flag_str.size() && flag_str[b] == '-') {
    // b is the index of the start of a flag.
    // Set e to the index just past the end of the flag.
    size_t e = b;
    while (e != flag_str.size() && isascii(flag_str[e]) &&
           (strchr("-_", flag_str[e]) != nullptr ||
            absl::ascii_isalnum(flag_str[e]))) {
      e++;
    }
    if (e != flag_str.size() && flag_str[e] == '=' &&
        e + 1 != flag_str.size() && strchr("'\"", flag_str[e + 1]) != nullptr) {
      // A flag of the form  --flag="something in double or single quotes"
      int c;
      e++;  // point just past '='
      size_t eflag = e;
      char quote = flag_str[e];
      e++;  // point just past quote
      // Put in value the string with quotes removed.
      std::string value;
      for (; e != flag_str.size() && (c = flag_str[e]) != quote; e++) {
        if (quote == '"' && c == '\\' && e + 1 != flag_str.size()) {
          // Handle backslash in double quoted strings.  They are literal in
          // single-quoted strings.
          e++;
          c = flag_str[e];
        }
        value += c;
      }
      if (e != flag_str.size()) {  // skip final " or '
        e++;
      }
      AppendToEnvArgv(flag_str.data() + b, eflag - b, value.data(),
                      value.size(), a);
    } else {  // A flag without a quoted value.
      e = FindFirstOf(flag_str, kWS, e);
      AppendToEnvArgv(flag_str.data() + b, e - b, "", 0, a);
    }
    b = FindFirstNotOf(flag_str, kWS, e);
  }
}

// Call ParseArgvFromString(..., a) on a string derived from the setting of the
// environment variable `envvar`, or a file it points to.
static void SetArgvFromEnv(absl::string_view envvar, EnvArgv* a) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("envvar: \"" + std::string(envvar.data(), envvar.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc mht_5(mht_5_v, 338, "", "./tensorflow/compiler/xla/parse_flags_from_env.cc", "SetArgvFromEnv");

  if (!a->initialized) {
    static const char kDummyArgv[] = "<argv[0]>";
    AppendToEnvArgv(kDummyArgv, strlen(kDummyArgv), nullptr, 0,
                    a);  // dummy argv[0]
    const char* env = getenv(std::string(envvar).c_str());
    if (env == nullptr || env[0] == '\0') {
      // nothing
    } else if (env[strspn(env, kWS)] == '-') {  // flags in env var value
      ParseArgvFromString(env, a);
    } else {  // assume it's a file name
      FILE* fp = fopen(env, "r");
      if (fp != nullptr) {
        std::string str;
        char buf[512];
        int n;
        while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
          str.append(buf, n);
        }
        fclose(fp);
        ParseArgvFromString(str, a);
      } else {
        LOG(QFATAL)
            << "Could not open file \"" << env
            << "\" to read flags for environment variable \"" << envvar
            << "\".  (We assumed \"" << env
            << "\" was a file name because it did not start with a \"--\".)";
      }
    }
    AppendToEnvArgv(nullptr, 0, nullptr, 0, a);  // add trailing nullptr to *a.
    a->initialized = true;
  }
}

// The simulated argv[] parsed from the environment, one for each different
// environment variable we've seen.
static absl::flat_hash_map<std::string, EnvArgv>& EnvArgvs() {
  static auto* env_argvs = new absl::flat_hash_map<std::string, EnvArgv>();
  return *env_argvs;
}

// Used to protect accesses to env_argvs.
static absl::Mutex env_argv_mu(absl::kConstInit);

bool ParseFlagsFromEnvAndDieIfUnknown(
    absl::string_view envvar, const std::vector<tensorflow::Flag>& flag_list) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("envvar: \"" + std::string(envvar.data(), envvar.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc mht_6(mht_6_v, 387, "", "./tensorflow/compiler/xla/parse_flags_from_env.cc", "ParseFlagsFromEnvAndDieIfUnknown");

  absl::MutexLock lock(&env_argv_mu);
  auto* env_argv = &EnvArgvs()[std::string(envvar)];
  SetArgvFromEnv(envvar, env_argv);  // a no-op if already initialized

  if (VLOG_IS_ON(1)) {
    VLOG(1) << "For env var " << envvar << " found arguments:";
    for (int i = 0; i < env_argv->argc; i++) {
      VLOG(1) << "  argv[" << i << "] = " << env_argv->argv[i];
    }
  }

  bool result =
      tensorflow::Flags::Parse(&env_argv->argc, &env_argv->argv[0], flag_list);

  // There's always at least one unparsed argc, namely the fake argv[0].
  if (result && env_argv->argc != 1) {
    // Skip the first argv, which is the fake argv[0].
    auto unknown_flags = absl::MakeSpan(env_argv->argv);
    unknown_flags.remove_prefix(1);

    // Some flags are set on XLA_FLAGS, others on TF_XLA_FLAGS.  If we find an
    // unrecognized flag, suggest the alternative.
    std::string alternate_envvar;
    if (envvar == "TF_XLA_FLAGS") {
      alternate_envvar = "XLA_FLAGS";
    } else if (envvar == "XLA_FLAGS") {
      alternate_envvar = "TF_XLA_FLAGS";
    }
    std::string did_you_mean;
    if (!alternate_envvar.empty()) {
      did_you_mean = absl::StrFormat(
          "\nPerhaps you meant to specify these on the %s envvar?",
          alternate_envvar);
    }

    LOG(QFATAL) << "Unknown flag" << (unknown_flags.size() > 1 ? "s" : "")
                << " in " << envvar << ": " << absl::StrJoin(unknown_flags, " ")
                << did_you_mean;
    return false;
  }
  return result;
}

// Testing only.
//
// Resets the env_argv struct so that subsequent calls to
// ParseFlagsFromEnvAndDieIfUnknown() will parse the environment variable (or
// the file it points to) anew, and set *pargc, and *pargv to point to the
// internal locations of the argc and argv constructed from the environment.
void ResetFlagsFromEnvForTesting(absl::string_view envvar, int** pargc,
                                 std::vector<char*>** pargv) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("envvar: \"" + std::string(envvar.data(), envvar.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSparse_flags_from_envDTcc mht_7(mht_7_v, 442, "", "./tensorflow/compiler/xla/parse_flags_from_env.cc", "ResetFlagsFromEnvForTesting");

  absl::MutexLock lock(&env_argv_mu);
  EnvArgvs().erase(std::string(envvar));
  auto& env_argv = EnvArgvs()[std::string(envvar)];
  *pargc = &env_argv.argc;
  *pargv = &env_argv.argv;
}

}  // namespace xla
