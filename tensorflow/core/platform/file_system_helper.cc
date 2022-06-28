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
class MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc() {
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

#include "tensorflow/core/platform/file_system_helper.h"

#include <deque>
#include <string>
#include <vector>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace internal {

namespace {

const int kNumThreads = port::NumSchedulableCPUs();

// Run a function in parallel using a ThreadPool, but skip the ThreadPool
// on the iOS platform due to its problems with more than a few threads.
void ForEach(int first, int last, const std::function<void(int)>& f) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc mht_0(mht_0_v, 210, "", "./tensorflow/core/platform/file_system_helper.cc", "ForEach");

#if TARGET_OS_IPHONE
  for (int i = first; i < last; i++) {
    f(i);
  }
#else
  int num_threads = std::min(kNumThreads, last - first);
  thread::ThreadPool threads(Env::Default(), "ForEach", num_threads);
  for (int i = first; i < last; i++) {
    threads.Schedule([f, i] { f(i); });
  }
#endif
}

// A globbing pattern can only start with these characters:
static const char kGlobbingChars[] = "*?[\\";

static inline bool IsGlobbingPattern(const std::string& pattern) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc mht_1(mht_1_v, 231, "", "./tensorflow/core/platform/file_system_helper.cc", "IsGlobbingPattern");

  return (pattern.find_first_of(kGlobbingChars) != std::string::npos);
}

// Make sure that the first entry in `dirs` during glob expansion does not
// contain a glob pattern. This is to prevent a corner-case bug where
// `<pattern>` would be treated differently than `./<pattern>`.
static std::string PatchPattern(const std::string& pattern) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc mht_2(mht_2_v, 242, "", "./tensorflow/core/platform/file_system_helper.cc", "PatchPattern");

  const std::string fixed_prefix =
      pattern.substr(0, pattern.find_first_of(kGlobbingChars));

  // Patching is needed when there is no directory part in `prefix`
  if (io::Dirname(fixed_prefix).empty()) {
    return io::JoinPath(".", pattern);
  }

  // No patching needed
  return pattern;
}

static std::vector<std::string> AllDirectoryPrefixes(const std::string& d) {
  std::vector<std::string> dirs;
  const std::string patched = PatchPattern(d);
  StringPiece dir(patched);

  // If the pattern ends with a `/` (or `\\` on Windows), we need to strip it
  // otherwise we would have one additional matching step and the result set
  // would be empty.
  bool is_directory = d[d.size() - 1] == '/';
#ifdef PLATFORM_WINDOWS
  is_directory = is_directory || (d[d.size() - 1] == '\\');
#endif
  if (is_directory) {
    dir = io::Dirname(dir);
  }

  while (!dir.empty()) {
    dirs.emplace_back(dir);
    StringPiece new_dir(io::Dirname(dir));
    // io::Dirname("/") returns "/" so we need to break the loop.
    // On Windows, io::Dirname("C:\\") would return "C:\\", so we check for
    // identity of the result instead of checking for dir[0] == `/`.
    if (dir == new_dir) break;
    dir = new_dir;
  }

  // Order the array from parent to ancestor (reverse order).
  std::reverse(dirs.begin(), dirs.end());

  return dirs;
}

static inline int GetFirstGlobbingEntry(const std::vector<std::string>& dirs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc mht_3(mht_3_v, 290, "", "./tensorflow/core/platform/file_system_helper.cc", "GetFirstGlobbingEntry");

  int i = 0;
  for (const auto& d : dirs) {
    if (IsGlobbingPattern(d)) {
      break;
    }
    i++;
  }
  return i;
}

}  // namespace

Status GetMatchingPaths(FileSystem* fs, Env* env, const string& pattern,
                        std::vector<string>* results) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("pattern: \"" + pattern + "\"");
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc mht_4(mht_4_v, 308, "", "./tensorflow/core/platform/file_system_helper.cc", "GetMatchingPaths");

  // Check that `fs`, `env` and `results` are non-null.
  if (fs == nullptr || env == nullptr || results == nullptr) {
    return Status(tensorflow::error::INVALID_ARGUMENT,
                  "Filesystem calls GetMatchingPaths with nullptr arguments");
  }

  // By design, we don't match anything on empty pattern
  results->clear();
  if (pattern.empty()) {
    return Status::OK();
  }

  // The pattern can contain globbing characters at multiple levels, e.g.:
  //
  //   foo/ba?/baz/f*r
  //
  // To match the full pattern, we must match every prefix subpattern and then
  // operate on the children for each match. Thus, we separate all subpatterns
  // in the `dirs` vector below.
  std::vector<std::string> dirs = AllDirectoryPrefixes(pattern);

  // We can have patterns that have several parents where no globbing is being
  // done, for example, `foo/bar/baz/*`. We don't need to expand the directories
  // which don't contain the globbing characters.
  int matching_index = GetFirstGlobbingEntry(dirs);

  // If we don't have globbing characters in the pattern then it specifies a
  // path in the filesystem. We add it to the result set if it exists.
  if (matching_index == dirs.size()) {
    if (fs->FileExists(pattern).ok()) {
      results->emplace_back(pattern);
    }
    return Status::OK();
  }

  // To expand the globbing, we do a BFS from `dirs[matching_index-1]`.
  // At every step, we work on a pair `{dir, ix}` such that `dir` is a real
  // directory, `ix < dirs.size() - 1` and `dirs[ix+1]` is a globbing pattern.
  // To expand the pattern, we select from all the children of `dir` only those
  // that match against `dirs[ix+1]`.
  // If there are more entries in `dirs` after `dirs[ix+1]` this mean we have
  // more patterns to match. So, we add to the queue only those children that
  // are also directories, paired with `ix+1`.
  // If there are no more entries in `dirs`, we return all children as part of
  // the answer.
  // Since we can get into a combinatorial explosion issue (e.g., pattern
  // `/*/*/*`), we process the queue in parallel. Each parallel processing takes
  // elements from `expand_queue` and adds them to `next_expand_queue`, after
  // which we swap these two queues (similar to double buffering algorithms).
  // PRECONDITION: `IsGlobbingPattern(dirs[0]) == false`
  // PRECONDITION: `matching_index > 0`
  // INVARIANT: If `{d, ix}` is in queue, then `d` and `dirs[ix]` are at the
  //            same level in the filesystem tree.
  // INVARIANT: If `{d, _}` is in queue, then `IsGlobbingPattern(d) == false`.
  // INVARIANT: If `{d, _}` is in queue, then `d` is a real directory.
  // INVARIANT: If `{_, ix}` is in queue, then `ix < dirs.size() - 1`.
  // INVARIANT: If `{_, ix}` is in queue, `IsGlobbingPattern(dirs[ix + 1])`.
  std::deque<std::pair<string, int>> expand_queue;
  std::deque<std::pair<string, int>> next_expand_queue;
  expand_queue.emplace_back(dirs[matching_index - 1], matching_index - 1);

  // Adding to `result` or `new_expand_queue` need to be protected by mutexes
  // since there are multiple threads writing to these.
  mutex result_mutex;
  mutex queue_mutex;

  while (!expand_queue.empty()) {
    next_expand_queue.clear();

    // The work item for every item in `expand_queue`.
    // pattern, we process them in parallel.
    auto handle_level = [&fs, &results, &dirs, &expand_queue,
                         &next_expand_queue, &result_mutex,
                         &queue_mutex](int i) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc mht_5(mht_5_v, 385, "", "./tensorflow/core/platform/file_system_helper.cc", "lambda");

      // See invariants above, all of these are valid accesses.
      const auto& queue_item = expand_queue.at(i);
      const std::string& parent = queue_item.first;
      const int index = queue_item.second + 1;
      const std::string& match_pattern = dirs[index];

      // Get all children of `parent`. If this fails, return early.
      std::vector<std::string> children;
      Status s = fs->GetChildren(parent, &children);
      if (s.code() == tensorflow::error::PERMISSION_DENIED) {
        return;
      }

      // Also return early if we don't have any children
      if (children.empty()) {
        return;
      }

      // Since we can get extremely many children here and on some filesystems
      // `IsDirectory` is expensive, we process the children in parallel.
      // We also check that children match the pattern in parallel, for speedup.
      // We store the status of the match and `IsDirectory` in
      // `children_status` array, one element for each children.
      std::vector<Status> children_status(children.size());
      auto handle_children = [&fs, &match_pattern, &parent, &children,
                              &children_status](int j) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfile_system_helperDTcc mht_6(mht_6_v, 414, "", "./tensorflow/core/platform/file_system_helper.cc", "lambda");

        const std::string path = io::JoinPath(parent, children[j]);
        if (!fs->Match(path, match_pattern)) {
          children_status[j] =
              Status(tensorflow::error::CANCELLED, "Operation not needed");
        } else {
          children_status[j] = fs->IsDirectory(path);
        }
      };
      ForEach(0, children.size(), handle_children);

      // At this point, pairing `children` with `children_status` will tell us
      // if a children:
      //   * does not match the pattern
      //   * matches the pattern and is a directory
      //   * matches the pattern and is not a directory
      // We fully ignore the first case.
      // If we matched the last pattern (`index == dirs.size() - 1`) then all
      // remaining children get added to the result.
      // Otherwise, only the directories get added to the next queue.
      for (size_t j = 0; j < children.size(); j++) {
        if (children_status[j].code() == tensorflow::error::CANCELLED) {
          continue;
        }

        const std::string path = io::JoinPath(parent, children[j]);
        if (index == dirs.size() - 1) {
          mutex_lock l(result_mutex);
          results->emplace_back(path);
        } else if (children_status[j].ok()) {
          mutex_lock l(queue_mutex);
          next_expand_queue.emplace_back(path, index);
        }
      }
    };
    ForEach(0, expand_queue.size(), handle_level);

    // After evaluating one level, swap the "buffers"
    std::swap(expand_queue, next_expand_queue);
  }

  return Status::OK();
}

}  // namespace internal
}  // namespace tensorflow
