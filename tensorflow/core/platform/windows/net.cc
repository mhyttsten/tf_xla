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
class MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSnetDTcc {
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
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSnetDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSnetDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/net.h"

#include <sys/types.h>
#include <winsock2.h>

#include <cstdlib>
#include <unordered_set>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/windows/error_windows.h"

#undef ERROR

namespace tensorflow {
namespace internal {

namespace {

bool IsPortAvailable(int* port, bool is_tcp) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSnetDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/platform/windows/net.cc", "IsPortAvailable");

  const int protocol = is_tcp ? IPPROTO_TCP : 0;
  SOCKET sock = socket(AF_INET, is_tcp ? SOCK_STREAM : SOCK_DGRAM, protocol);

  struct sockaddr_in addr;
  int addr_len = static_cast<int>(sizeof(addr));
  int actual_port;

  CHECK_GE(*port, 0);
  CHECK_LE(*port, 65535);
  if (sock == INVALID_SOCKET) {
    LOG(ERROR) << "socket() failed: " << WindowsWSAGetLastErrorMessage();
    return false;
  }

  // SO_REUSEADDR lets us start up a server immediately after it exits.
  const int one = 1;
  int result = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                          reinterpret_cast<const char*>(&one), sizeof(one));
  if (result == SOCKET_ERROR) {
    LOG(ERROR) << "setsockopt() failed: " << WindowsWSAGetLastErrorMessage();
    closesocket(sock);
    return false;
  }

  // Try binding to port.
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons((uint16_t)*port);
  result = bind(sock, (struct sockaddr*)&addr, sizeof(addr));
  if (result == SOCKET_ERROR) {
    LOG(WARNING) << "bind(port=" << *port
                 << ") failed: " << WindowsWSAGetLastErrorMessage();
    closesocket(sock);
    return false;
  }

  // Get the bound port number.
  result = getsockname(sock, (struct sockaddr*)&addr, &addr_len);
  if (result == SOCKET_ERROR) {
    LOG(WARNING) << "getsockname() failed: " << WindowsWSAGetLastErrorMessage();
    closesocket(sock);
    return false;
  }

  CHECK_LE(addr_len, sizeof(addr));
  actual_port = ntohs(addr.sin_port);
  CHECK_GT(actual_port, 0);
  if (*port == 0) {
    *port = actual_port;
  } else {
    CHECK_EQ(*port, actual_port);
  }

  closesocket(sock);
  return true;
}

const int kNumRandomPortsToPick = 100;
const int kMaximumTrials = 1000;

}  // namespace

int PickUnusedPortOrDie() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSwindowsPSnetDTcc mht_1(mht_1_v, 270, "", "./tensorflow/core/platform/windows/net.cc", "PickUnusedPortOrDie");

  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != NO_ERROR) {
    LOG(ERROR) << "Error at WSAStartup()";
    return false;
  }

  static std::unordered_set<int> chosen_ports;

  // Type of port to first pick in the next iteration.
  bool is_tcp = true;
  int trial = 0;
  while (true) {
    int port;
    trial++;
    CHECK_LE(trial, kMaximumTrials)
        << "Failed to pick an unused port for testing.";
    if (trial == 1) {
      port = GetCurrentProcessId() % (65536 - 30000) + 30000;
    } else if (trial <= kNumRandomPortsToPick) {
      port = rand() % (65536 - 30000) + 30000;
    } else {
      port = 0;
    }

    if (chosen_ports.find(port) != chosen_ports.end()) {
      continue;
    }
    if (!IsPortAvailable(&port, is_tcp)) {
      continue;
    }

    CHECK_GT(port, 0);
    if (!IsPortAvailable(&port, !is_tcp)) {
      is_tcp = !is_tcp;
      continue;
    }

    chosen_ports.insert(port);
    WSACleanup();
    return port;
  }

  return 0;
}

}  // namespace internal
}  // namespace tensorflow
