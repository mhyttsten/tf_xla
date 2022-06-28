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
class MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc {
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
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc() {
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

#include "tensorflow/core/util/device_name_utils.h"

#include <algorithm>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

static bool IsAlpha(char c) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/util/device_name_utils.cc", "IsAlpha");

  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool IsAlphaNumOrUnderscore(char c) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("c: '" + std::string(1, c) + "'");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_1(mht_1_v, 205, "", "./tensorflow/core/util/device_name_utils.cc", "IsAlphaNumOrUnderscore");

  return IsAlpha(c) || (c >= '0' && c <= '9') || c == '_';
}

// Returns true iff "in" is a valid job name.
static bool IsJobName(StringPiece in) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_2(mht_2_v, 213, "", "./tensorflow/core/util/device_name_utils.cc", "IsJobName");

  return !in.empty() && IsAlpha(in.front()) &&
         std::all_of(in.begin(), in.end(), IsAlphaNumOrUnderscore);
}

static bool ConsumePrefix(StringPiece* in, string* out,
                          StringPiece prefix_terminators) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_3(mht_3_v, 222, "", "./tensorflow/core/util/device_name_utils.cc", "ConsumePrefix");

  if (in->empty() || !IsAlpha(in->front())) return false;
  const auto end_it =
      std::find_first_of(in->begin(), in->end(), prefix_terminators.begin(),
                         prefix_terminators.end());
  if (!std::all_of(in->begin(), end_it, IsAlphaNumOrUnderscore)) {
    return false;
  }
  out->assign(in->begin(), end_it);
  in->remove_prefix(end_it - in->begin());
  return true;
}

// Returns true and fills in "*job" iff "*in" starts with a job name.
static bool ConsumeJobName(StringPiece* in, string* job) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_4(mht_4_v, 239, "", "./tensorflow/core/util/device_name_utils.cc", "ConsumeJobName");

  return ConsumePrefix(in, job, "/");
}

// Returns true and fills in "*device_type" iff "*in" starts with a device type
// name.
static bool ConsumeDeviceType(StringPiece* in, string* device_type) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_5(mht_5_v, 248, "", "./tensorflow/core/util/device_name_utils.cc", "ConsumeDeviceType");

  return ConsumePrefix(in, device_type, "/:");
}

// Returns true and fills in "*val" iff "*in" starts with a decimal
// number.
static bool ConsumeNumber(StringPiece* in, int* val) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_6(mht_6_v, 257, "", "./tensorflow/core/util/device_name_utils.cc", "ConsumeNumber");

  uint64 tmp;
  if (str_util::ConsumeLeadingDigits(in, &tmp)) {
    *val = tmp;
    return true;
  } else {
    return false;
  }
}

// Returns a fully qualified device name given the parameters.
static string DeviceName(const string& job, int replica, int task,
                         const string& device_prefix, const string& device_type,
                         int id) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("job: \"" + job + "\"");
   mht_7_v.push_back("device_prefix: \"" + device_prefix + "\"");
   mht_7_v.push_back("device_type: \"" + device_type + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_7(mht_7_v, 276, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceName");

  CHECK(IsJobName(job)) << job;
  CHECK_LE(0, replica);
  CHECK_LE(0, task);
  CHECK(!device_type.empty());
  CHECK_LE(0, id);
  return strings::StrCat("/job:", job, "/replica:", replica, "/task:", task,
                         device_prefix, device_type, ":", id);
}

/* static */
string DeviceNameUtils::FullName(const string& job, int replica, int task,
                                 const string& type, int id) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("job: \"" + job + "\"");
   mht_8_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_8(mht_8_v, 293, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::FullName");

  return DeviceName(job, replica, task, "/device:", type, id);
}

namespace {
string LegacyName(const string& job, int replica, int task, const string& type,
                  int id) {
   std::vector<std::string> mht_9_v;
   mht_9_v.push_back("job: \"" + job + "\"");
   mht_9_v.push_back("type: \"" + type + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_9(mht_9_v, 304, "", "./tensorflow/core/util/device_name_utils.cc", "LegacyName");

  return DeviceName(job, replica, task, "/", absl::AsciiStrToLower(type), id);
}
}  // anonymous namespace

bool DeviceNameUtils::ParseFullName(StringPiece fullname, ParsedName* p) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_10(mht_10_v, 312, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::ParseFullName");

  p->Clear();
  if (fullname == "/") {
    return true;
  }
  while (!fullname.empty()) {
    bool progress = false;
    if (absl::ConsumePrefix(&fullname, "/job:")) {
      p->has_job = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_job && !ConsumeJobName(&fullname, &p->job)) {
        return false;
      }
      progress = true;
    }
    if (absl::ConsumePrefix(&fullname, "/replica:")) {
      p->has_replica = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_replica && !ConsumeNumber(&fullname, &p->replica)) {
        return false;
      }
      progress = true;
    }
    if (absl::ConsumePrefix(&fullname, "/task:")) {
      p->has_task = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_task && !ConsumeNumber(&fullname, &p->task)) {
        return false;
      }
      progress = true;
    }
    if (absl::ConsumePrefix(&fullname, "/device:")) {
      p->has_type = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_type && !ConsumeDeviceType(&fullname, &p->type)) {
        return false;
      }
      if (!absl::ConsumePrefix(&fullname, ":")) {
        p->has_id = false;
      } else {
        p->has_id = !absl::ConsumePrefix(&fullname, "*");
        if (p->has_id && !ConsumeNumber(&fullname, &p->id)) {
          return false;
        }
      }
      progress = true;
    }

    // Handle legacy naming convention for cpu and gpu.
    if (absl::ConsumePrefix(&fullname, "/cpu:") ||
        absl::ConsumePrefix(&fullname, "/CPU:")) {
      p->has_type = true;
      p->type = "CPU";  // Treat '/cpu:..' as uppercase '/device:CPU:...'
      p->has_id = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_id && !ConsumeNumber(&fullname, &p->id)) {
        return false;
      }
      progress = true;
    }
    if (absl::ConsumePrefix(&fullname, "/gpu:") ||
        absl::ConsumePrefix(&fullname, "/GPU:")) {
      p->has_type = true;
      p->type = "GPU";  // Treat '/gpu:..' as uppercase '/device:GPU:...'
      p->has_id = !absl::ConsumePrefix(&fullname, "*");
      if (p->has_id && !ConsumeNumber(&fullname, &p->id)) {
        return false;
      }
      progress = true;
    }

    if (!progress) {
      return false;
    }
  }
  return true;
}

bool DeviceNameUtils::ParseFullOrLocalName(StringPiece fullname,
                                           ParsedName* p) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_11(mht_11_v, 389, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::ParseFullOrLocalName");

  return ParseFullName(fullname, p) || ParseLocalName(fullname, p);
}

namespace {

void CompleteName(const DeviceNameUtils::ParsedName& parsed_basename,
                  DeviceNameUtils::ParsedName* parsed_name) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_12(mht_12_v, 399, "", "./tensorflow/core/util/device_name_utils.cc", "CompleteName");

  if (!parsed_name->has_job) {
    parsed_name->job = parsed_basename.job;
    parsed_name->has_job = true;
  }
  if (!parsed_name->has_replica) {
    parsed_name->replica = parsed_basename.replica;
    parsed_name->has_replica = true;
  }
  if (!parsed_name->has_task) {
    parsed_name->task = parsed_basename.task;
    parsed_name->has_task = true;
  }
  if (!parsed_name->has_type) {
    parsed_name->type = parsed_basename.type;
    parsed_name->has_type = true;
  }
  if (!parsed_name->has_id) {
    parsed_name->id = parsed_basename.id;
    parsed_name->has_id = true;
  }
}

}  // namespace

/* static */
Status DeviceNameUtils::CanonicalizeDeviceName(StringPiece fullname,
                                               StringPiece basename,
                                               string* canonical_name) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_13(mht_13_v, 430, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::CanonicalizeDeviceName");

  *canonical_name = "";
  ParsedName parsed_basename;
  if (!ParseFullName(basename, &parsed_basename)) {
    return errors::InvalidArgument("Could not parse basename: ", basename,
                                   " into a device specification.");
  }
  if (!(parsed_basename.has_job && parsed_basename.has_replica &&
        parsed_basename.has_task && parsed_basename.has_type &&
        parsed_basename.has_id)) {
    return errors::InvalidArgument("Basename: ", basename,
                                   " should be fully "
                                   "specified.");
  }
  ParsedName parsed_name;
  if (ParseLocalName(fullname, &parsed_name)) {
    CompleteName(parsed_basename, &parsed_name);
    *canonical_name = ParsedNameToString(parsed_name);
    return Status::OK();
  }
  if (ParseFullName(fullname, &parsed_name)) {
    CompleteName(parsed_basename, &parsed_name);
    *canonical_name = ParsedNameToString(parsed_name);
    return Status::OK();
  }
  return errors::InvalidArgument("Could not parse ", fullname,
                                 " into a device "
                                 "specification.");
}

/* static */
string DeviceNameUtils::ParsedNameToString(const ParsedName& pn) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_14(mht_14_v, 464, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::ParsedNameToString");

  string buf;
  if (pn.has_job) strings::StrAppend(&buf, "/job:", pn.job);
  if (pn.has_replica) strings::StrAppend(&buf, "/replica:", pn.replica);
  if (pn.has_task) strings::StrAppend(&buf, "/task:", pn.task);
  if (pn.has_type) {
    strings::StrAppend(&buf, "/device:", pn.type, ":");
    if (pn.has_id) {
      strings::StrAppend(&buf, pn.id);
    } else {
      strings::StrAppend(&buf, "*");
    }
  }
  return buf;
}

/* static */
bool DeviceNameUtils::IsSpecification(const ParsedName& less_specific,
                                      const ParsedName& more_specific) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_15(mht_15_v, 485, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::IsSpecification");

  if (less_specific.has_job &&
      (!more_specific.has_job || (less_specific.job != more_specific.job))) {
    return false;
  }
  if (less_specific.has_replica &&
      (!more_specific.has_replica ||
       (less_specific.replica != more_specific.replica))) {
    return false;
  }
  if (less_specific.has_task &&
      (!more_specific.has_task || (less_specific.task != more_specific.task))) {
    return false;
  }
  if (less_specific.has_type &&
      (!more_specific.has_type || (less_specific.type != more_specific.type))) {
    return false;
  }
  if (less_specific.has_id &&
      (!more_specific.has_id || (less_specific.id != more_specific.id))) {
    return false;
  }
  return true;
}

/* static */
bool DeviceNameUtils::AreCompatibleDevNames(const ParsedName& a,
                                            const ParsedName& b) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_16(mht_16_v, 515, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::AreCompatibleDevNames");

  if (a.has_job && b.has_job && (a.job != b.job)) {
    return false;
  }
  if (a.has_replica && b.has_replica && (a.replica != b.replica)) {
    return false;
  }
  if (a.has_task && b.has_task && (a.task != b.task)) {
    return false;
  }
  if (a.has_type && b.has_type && (a.type != b.type)) {
    return false;
  }
  if (a.has_id && b.has_id && (a.id != b.id)) {
    return false;
  }
  return true;
}

void DeviceNameUtils::EnsureSpecification(ParsedName* more_specific,
                                          const ParsedName& less_specific) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_17(mht_17_v, 538, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::EnsureSpecification");

  if (less_specific.has_job) {
    more_specific->has_job = true;
    more_specific->job = less_specific.job;
  }
  if (less_specific.has_replica) {
    more_specific->has_replica = true;
    more_specific->replica = less_specific.replica;
  }
  if (less_specific.has_task) {
    more_specific->has_task = true;
    more_specific->task = less_specific.task;
  }
  if (less_specific.has_type) {
    more_specific->has_type = true;
    more_specific->type = less_specific.type;
  }
  if (less_specific.has_id) {
    more_specific->has_id = true;
    more_specific->id = less_specific.id;
  }
}

/* static */
bool DeviceNameUtils::IsCompleteSpecification(const ParsedName& pattern,
                                              const ParsedName& name) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_18(mht_18_v, 566, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::IsCompleteSpecification");

  CHECK(name.has_job && name.has_replica && name.has_task && name.has_type &&
        name.has_id);

  if (pattern.has_job && (pattern.job != name.job)) return false;
  if (pattern.has_replica && (pattern.replica != name.replica)) return false;
  if (pattern.has_task && (pattern.task != name.task)) return false;
  if (pattern.has_type && (pattern.type != name.type)) return false;
  if (pattern.has_id && (pattern.id != name.id)) return false;
  return true;
}

namespace {
Status MergeDevNamesImpl(DeviceNameUtils::ParsedName* target,
                         const DeviceNameUtils::ParsedName& other,
                         bool allow_soft_placement, bool override_conflicts) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_19(mht_19_v, 584, "", "./tensorflow/core/util/device_name_utils.cc", "MergeDevNamesImpl");

  const auto& ParsedNameToString = DeviceNameUtils::ParsedNameToString;
  if (other.has_job) {
    if (target->has_job && target->job != other.job) {
      return errors::InvalidArgument(
          "Cannot merge devices with incompatible jobs: '",
          ParsedNameToString(*target), "' and '", ParsedNameToString(other),
          "'");
    } else {
      target->has_job = other.has_job;
      target->job = other.job;
    }
  }

  if (other.has_replica) {
    if (target->has_replica && target->replica != other.replica) {
      return errors::InvalidArgument(
          "Cannot merge devices with incompatible replicas: '",
          ParsedNameToString(*target), "' and '", ParsedNameToString(other),
          "'");
    } else {
      target->has_replica = other.has_replica;
      target->replica = other.replica;
    }
  }

  if (other.has_task) {
    if (target->has_task && target->task != other.task) {
      return errors::InvalidArgument(
          "Cannot merge devices with incompatible tasks: '",
          ParsedNameToString(*target), "' and '", ParsedNameToString(other),
          "'");
    } else {
      target->has_task = other.has_task;
      target->task = other.task;
    }
  }

  if (other.has_type) {
    if (target->has_type && target->type != other.type) {
      if (!allow_soft_placement) {
        return errors::InvalidArgument(
            "Cannot merge devices with incompatible types: '",
            ParsedNameToString(*target), "' and '", ParsedNameToString(other),
            "'");
      } else if (override_conflicts) {
        target->type = other.type;
      } else {
        target->has_id = false;
        target->has_type = false;
        return Status::OK();
      }
    } else {
      target->has_type = other.has_type;
      target->type = other.type;
    }
  }

  if (other.has_id) {
    if (target->has_id && target->id != other.id) {
      if (!allow_soft_placement) {
        return errors::InvalidArgument(
            "Cannot merge devices with incompatible ids: '",
            ParsedNameToString(*target), "' and '", ParsedNameToString(other),
            "'");
      } else if (override_conflicts) {
        target->id = other.id;
      } else {
        target->has_id = false;
        return Status::OK();
      }
    } else {
      target->has_id = other.has_id;
      target->id = other.id;
    }
  }

  return Status::OK();
}

}  // namespace

/* static */
Status DeviceNameUtils::MergeDevNames(ParsedName* target,
                                      const ParsedName& other,
                                      bool allow_soft_placement) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_20(mht_20_v, 672, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::MergeDevNames");

  return MergeDevNamesImpl(target, other, allow_soft_placement,
                           /*override_conflicts=*/false);
}

/* static */
Status DeviceNameUtils::MergeOverrideDevNames(ParsedName* target,
                                              const ParsedName& other) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_21(mht_21_v, 682, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::MergeOverrideDevNames");

  return MergeDevNamesImpl(target, other, /*allow_soft_placement=*/true,
                           /*override_conflicts=*/true);
}

/* static */
void DeviceNameUtils::MergeUnsetDevNames(ParsedName* target,
                                         const ParsedName& other) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_22(mht_22_v, 692, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::MergeUnsetDevNames");

  if (other.has_job && !target->has_job) {
    target->has_job = other.has_job;
    target->job = other.job;
  }

  if (other.has_replica && !target->has_replica) {
    target->has_replica = other.has_replica;
    target->replica = other.replica;
  }

  if (other.has_task && !target->has_task) {
    target->has_task = other.has_task;
    target->task = other.task;
  }

  if (other.has_type && !target->has_type) {
    target->has_type = other.has_type;
    target->type = other.type;
  }

  if (other.has_id && !target->has_id) {
    target->has_id = other.has_id;
    target->id = other.id;
  }
}

/* static */
bool DeviceNameUtils::IsSameAddressSpace(const ParsedName& a,
                                         const ParsedName& b) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_23(mht_23_v, 724, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::IsSameAddressSpace");

  return (a.has_job && b.has_job && (a.job == b.job)) &&
         (a.has_replica && b.has_replica && (a.replica == b.replica)) &&
         (a.has_task && b.has_task && (a.task == b.task));
}

/* static */
bool DeviceNameUtils::IsSameAddressSpace(StringPiece src, StringPiece dst) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_24(mht_24_v, 734, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::IsSameAddressSpace");

  ParsedName x;
  ParsedName y;
  return ParseFullName(src, &x) && ParseFullName(dst, &y) &&
         IsSameAddressSpace(x, y);
}

/* static */
bool DeviceNameUtils::IsDifferentAddressSpace(const ParsedName& a,
                                              const ParsedName& b) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_25(mht_25_v, 746, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::IsDifferentAddressSpace");

  return (a.has_job && b.has_job && (a.job != b.job)) ||
         (a.has_replica && b.has_replica && (a.replica != b.replica)) ||
         (a.has_task && b.has_task && (a.task != b.task));
}

/* static */
const DeviceNameUtils::ParsedName DeviceNameUtils::AddressSpace(
    const ParsedName& name) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_26(mht_26_v, 757, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::AddressSpace");

  ParsedName address_space;
  address_space.has_job = name.has_job;
  address_space.has_replica = name.has_replica;
  address_space.has_task = name.has_task;
  address_space.job = name.job;
  address_space.replica = name.replica;
  address_space.task = name.task;
  return address_space;
}

/* static */
string DeviceNameUtils::LocalName(StringPiece type, int id) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_27(mht_27_v, 772, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::LocalName");

  return strings::StrCat("/device:", type, ":", id);
}

namespace {
// Returns the legacy local device name given its "type" and "id" (which is
// '/device:type:id').
string LegacyLocalName(StringPiece type, int id) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_28(mht_28_v, 782, "", "./tensorflow/core/util/device_name_utils.cc", "LegacyLocalName");

  return strings::StrCat(type, ":", id);
}
}  // anonymous namespace

/* static */
string DeviceNameUtils::LocalName(StringPiece fullname) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_29(mht_29_v, 791, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::LocalName");

  ParsedName x;
  CHECK(ParseFullName(fullname, &x)) << fullname;
  return LocalName(x.type, x.id);
}

/* static */
bool DeviceNameUtils::ParseLocalName(StringPiece name, ParsedName* p) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_30(mht_30_v, 801, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::ParseLocalName");

  if (!ConsumeDeviceType(&name, &p->type)) {
    return false;
  }
  p->has_type = true;
  if (!absl::ConsumePrefix(&name, ":")) {
    return false;
  }
  if (!ConsumeNumber(&name, &p->id)) {
    return false;
  }
  p->has_id = true;
  return name.empty();
}

/* static */
bool DeviceNameUtils::SplitDeviceName(StringPiece name, string* task,
                                      string* device) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_31(mht_31_v, 821, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::SplitDeviceName");

  ParsedName pn;
  if (ParseFullName(name, &pn) && pn.has_type && pn.has_id) {
    task->clear();
    task->reserve(
        (pn.has_job ? (5 + pn.job.size()) : 0) +
        (pn.has_replica ? (9 + 4 /*estimated UB for # replica digits*/) : 0) +
        (pn.has_task ? (6 + 4 /*estimated UB for # task digits*/) : 0));
    if (pn.has_job) {
      strings::StrAppend(task, "/job:", pn.job);
    }
    if (pn.has_replica) {
      strings::StrAppend(task, "/replica:", pn.replica);
    }
    if (pn.has_task) {
      strings::StrAppend(task, "/task:", pn.task);
    }
    device->clear();
    strings::StrAppend(device, pn.type, ":", pn.id);
    return true;
  }
  return false;
}

/* static */
bool DeviceNameUtils::GetTaskName(const ParsedName& pn, string* task) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_32(mht_32_v, 849, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::GetTaskName");

  if (pn.has_job && pn.has_replica && pn.has_task) {
    task->clear();
    task->reserve((5 + pn.job.size()) +
                  (9 + 4 /*estimated UB for # replica digits*/) +
                  (6 + 4 /*estimated UB for # task digits*/));
    strings::StrAppend(task, "/job:", pn.job);
    strings::StrAppend(task, "/replica:", pn.replica);
    strings::StrAppend(task, "/task:", pn.task);
    return true;
  }
  return false;
}

std::vector<string> DeviceNameUtils::GetNamesForDeviceMappings(
    const ParsedName& pn) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_33(mht_33_v, 867, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::GetNamesForDeviceMappings");

  if (pn.has_job && pn.has_replica && pn.has_task && pn.has_type && pn.has_id) {
    return {
        DeviceNameUtils::FullName(pn.job, pn.replica, pn.task, pn.type, pn.id),
        LegacyName(pn.job, pn.replica, pn.task, pn.type, pn.id)};
  } else {
    return {};
  }
}

std::vector<string> DeviceNameUtils::GetLocalNamesForDeviceMappings(
    const ParsedName& pn) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_34(mht_34_v, 881, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::GetLocalNamesForDeviceMappings");

  if (pn.has_type && pn.has_id) {
    return {DeviceNameUtils::LocalName(pn.type, pn.id),
            LegacyLocalName(pn.type, pn.id)};
  } else {
    return {};
  }
}

/*static*/ Status DeviceNameUtils::DeviceNameToCpuDeviceName(
    const string& device_name, string* host_device_name) {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_35(mht_35_v, 895, "", "./tensorflow/core/util/device_name_utils.cc", "DeviceNameUtils::DeviceNameToCpuDeviceName");

  DeviceNameUtils::ParsedName device;
  if (!DeviceNameUtils::ParseFullName(device_name, &device)) {
    return errors::Internal("Could not parse device name ", device_name);
  }
  device.type = "CPU";
  device.has_type = true;
  device.id = 0;
  device.has_id = true;
  *host_device_name = DeviceNameUtils::ParsedNameToString(device);
  return Status::OK();
}

std::ostream& operator<<(std::ostream& os,
                         const DeviceNameUtils::ParsedName& x) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTcc mht_36(mht_36_v, 912, "", "./tensorflow/core/util/device_name_utils.cc", "operator<<");

  os << DeviceNameUtils::ParsedNameToString(x);
  return os;
}

}  // namespace tensorflow
