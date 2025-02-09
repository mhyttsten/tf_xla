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

#ifndef TENSORFLOW_CORE_UTIL_DEVICE_NAME_UTILS_H_
#define TENSORFLOW_CORE_UTIL_DEVICE_NAME_UTILS_H_
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
class MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTh {
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
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTh() {
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


#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

// In TensorFlow a device name is a string of the following form:
//   /job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num>
//
// <name> is a short identifier conforming to the regexp
//     [a-zA-Z][_a-zA-Z]*
// <type> is a supported device type (e.g. 'cpu' or 'gpu')
// <replica>, <task>, <device_num> are small non-negative integers and are
// densely allocated (except in tests).
//
// For some purposes, we also allow device patterns, which can specify
// some or none of the specific fields above, with missing components,
// or "<component>:*" indicating "any value allowed for that component.
//
// For example:
//   "/job:param_server"   - Consider any devices in the "param_server" job
//   "/device:cpu:*"       - Consider any cpu devices in any job/task/replica
//   "/job:*/replica:*/task:*/device:cpu:*"  - Consider any cpu devices in any
//                                             job/task/replica
//   "/job:w/replica:0/task:0/device:gpu:*"  - Consider any gpu devices in
//                                             replica 0, task 0, of job "w"
class DeviceNameUtils {
 public:
  // Returns a fully qualified device name given the parameters.
  static std::string FullName(const std::string& job, int replica, int task,
                              const std::string& type, int id);

  struct ParsedName {
    void Clear() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTh mht_0(mht_0_v, 222, "", "./tensorflow/core/util/device_name_utils.h", "Clear");

      has_job = false;
      has_replica = false;
      has_task = false;
      has_type = false;
      has_id = false;
      job.clear();
      replica = 0;
      task = 0;
      type.clear();
      id = 0;
    }

    bool operator==(const ParsedName& other) const {
      return (has_job ? (other.has_job && job == other.job) : !other.has_job) &&
             (has_replica ? (other.has_replica && replica == other.replica)
                          : !other.has_replica) &&
             (has_task ? (other.has_task && task == other.task)
                       : !other.has_task) &&
             (has_type ? (other.has_type && type == other.type)
                       : !other.has_type) &&
             (has_id ? (other.has_id && id == other.id) : !other.has_id);
    }

    bool operator!=(const ParsedName& other) const {
      return !operator==(other);
    }

    bool has_job = false;
    std::string job;
    bool has_replica = false;
    int replica = 0;
    bool has_task = false;
    int task = 0;
    bool has_type = false;
    std::string type;
    bool has_id = false;
    int id = 0;
  };

  // Parses the device name, first as a full name, then, if it fails, as a
  // global one. Returns `false` if both attempts fail.
  static bool ParseFullOrLocalName(StringPiece fullname, ParsedName* parsed);

  // Parses "fullname" into "*parsed". Returns true iff succeeds.
  // Legacy names like "/cpu:0" that don't contain "device",
  // are parsed to mean their current counterparts "/device:CPU:0". More
  // specifically, the lower case "cpu" and "gpu" is capitalized and "device"
  // is added. "/tpu:0" is not treated the same way - it has use the current
  // full syntax.
  // Also, note that lower case "cpu" and "gpu" device types in current syntax
  // are not capitalized. For example, "/device:CPU:0" is different from
  // "/device:cpu:0"
  static bool ParseFullName(StringPiece fullname, ParsedName* parsed);

  // Canonicalizes "fullname" into "*canonical_name". Uses a fully specified
  // basename to fill in fields that are missing. Accepts both legacy, newer
  // and local versions of the device spec. Returns the newer version of the
  // device spec. If we were unable to interpret / parse "fullname" returns
  // an error and *canonical_name is set to "".
  static Status CanonicalizeDeviceName(StringPiece fullname,
                                       StringPiece basename,
                                       std::string* canonical_name);

  // Returns true if "name" specifies any non-trivial constraint on the device.
  static bool HasSomeDetails(const ParsedName& name) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTh mht_1(mht_1_v, 290, "", "./tensorflow/core/util/device_name_utils.h", "HasSomeDetails");

    return name.has_job || name.has_replica || name.has_task || name.has_type ||
           name.has_id;
  }

  // Returns true if more_specific is a specification of
  // less_specific, i.e. everywhere that less-specific has a
  // non-wildcard component value, more_specific has the same value
  // for that component.
  static bool IsSpecification(const ParsedName& less_specific,
                              const ParsedName& more_specific);

  // Makes minimal changes to more_specific so that it becomes a
  // specification of less_specific.
  static void EnsureSpecification(ParsedName* more_specific,
                                  const ParsedName& less_specific);

  // Like IsSpecification, but the second argument "name" must have a
  // non-wildcard value for all of its components.
  static bool IsCompleteSpecification(const ParsedName& pattern,
                                      const ParsedName& name);

  // True iff there exists any possible device name that is a specification of
  // both "a" and "b".
  static bool AreCompatibleDevNames(const ParsedName& a, const ParsedName& b);

  // Merges the device specifications in "*target" and "other", and
  // stores the result in "*target". Returns OK if "*target" and
  // "other" are compatible, otherwise returns an error.
  static Status MergeDevNames(ParsedName* target, const ParsedName& other) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSutilPSdevice_name_utilsDTh mht_2(mht_2_v, 322, "", "./tensorflow/core/util/device_name_utils.h", "MergeDevNames");

    return MergeDevNames(target, other, false);
  }
  static Status MergeDevNames(ParsedName* target, const ParsedName& other,
                              bool allow_soft_placement);
  // Same as MergeDevNames with allow_soft_placement=true, but instead of
  // clearing conflicting fields, overrides them with `other`'s values.
  static Status MergeOverrideDevNames(ParsedName* target,
                                      const ParsedName& other);

  // Merges the device specifications in "*target" and "other", and
  // stores the result in "*target" by setting all unset values in target with
  // corresponding set ones in other.
  static void MergeUnsetDevNames(ParsedName* target, const ParsedName& other);

  // Returns true iff devices identified by 'src' and 'dst' are in the
  // same address space.
  static bool IsSameAddressSpace(StringPiece src, StringPiece dst);
  static bool IsSameAddressSpace(const ParsedName& src, const ParsedName& dst);

  // Returns true iff devices identified by 'a' and 'b' are in different
  // address space.
  static bool IsDifferentAddressSpace(const ParsedName& a, const ParsedName& b);

  // Returns the an address space specification containing only the
  // job/replica/task of the given name.
  static const ParsedName AddressSpace(const ParsedName& name);

  // Returns the local device given its "type" and "id".
  static std::string LocalName(StringPiece type, int id);

  // Returns a short local device name (cpu:0, gpu:1, etc) based on
  // the given fullname.
  static std::string LocalName(StringPiece fullname);

  // If "name" is a valid local device name (cpu:0, gpu:1, etc.),
  // fills in parsed.type and parsed.id accordingly. Returns true iff
  // succeeds.
  static bool ParseLocalName(StringPiece name, ParsedName* parsed);

  // Splits a fully-qualified device name into a task identifier and a
  // relative device identifier. It first parses "name" using
  // ParseFullName(), then assigns *task with everything except for
  // the local device component, and assigns the relative device
  // component into *device.  This function will still return true if
  // the task component is empty, but it requires the relative device
  // component to be fully specified.
  static bool SplitDeviceName(StringPiece name, std::string* task,
                              std::string* device);

  // Get the task name from ParsedName. Return false if the task component is
  // not fully specified.
  static bool GetTaskName(const ParsedName& pn, std::string* task);

  static std::string ParsedNameToString(const ParsedName& pn);

  // Returns canonical and legacy full names for the given parsed
  // device name 'pn'. The returned string names are often useful to
  // look up devices from a mapping.
  static std::vector<string> GetNamesForDeviceMappings(const ParsedName& pn);

  // Returns canonical and legacy local names for the given parsed device name
  // 'pn'. The returned string names are often useful to look up devices from a
  // mapping.
  static std::vector<string> GetLocalNamesForDeviceMappings(
      const ParsedName& pn);

  // Returns name of the CPU:0 device on the same host as the device
  // `device_name`.
  static Status DeviceNameToCpuDeviceName(const std::string& device_name,
                                          std::string* host_device_name);
};

std::ostream& operator<<(std::ostream& os,
                         const DeviceNameUtils::ParsedName& x);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_DEVICE_NAME_UTILS_H_
