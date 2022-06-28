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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc() {
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
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstring>

#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/soc_model.h"

namespace {

void* LoadLibadsprpc() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "LoadLibadsprpc");

  void* lib = dlopen("libadsprpc.so", RTLD_LAZY | RTLD_LOCAL);
  if (lib) {
    fprintf(stdout, "loaded libadsprpc.so\n");
    return lib;
  }

  return nullptr;
}

void* LoadLibcdsprpc() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_1(mht_1_v, 210, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "LoadLibcdsprpc");

  void* lib = dlopen("libcdsprpc.so", RTLD_LAZY | RTLD_LOCAL);
  if (lib) {
    fprintf(stdout, "loaded libcdsprpc.so\n");
    return lib;
  }

  return nullptr;
}

void* LoadDsprpc() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_2(mht_2_v, 223, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "LoadDsprpc");

  SocSkelTable soc_model = tflite::delegates::getsoc_model();
  // Use aDSP for 835 and 820, otherwise cDSP.
  if (soc_model.mode == NON_DOMAINS ||
      (soc_model.dsp_type != nullptr &&
       strcmp(soc_model.dsp_type, "adsp") == 0)) {
    return LoadLibadsprpc();
  }
  return LoadLibcdsprpc();
}

void* LoadFunction(const char* name) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_3(mht_3_v, 238, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "LoadFunction");

  static void* libadsprpc = LoadDsprpc();
  if (libadsprpc == nullptr) {
    fprintf(stderr, "libadsprpc handle is NULL\n");
    return nullptr;
  }
  auto* func_pt = dlsym(libadsprpc, name);
  if (func_pt == nullptr) {
    fprintf(stderr, "Func %s not available on this device (NULL).\n", name);
  }
  return func_pt;
}

using remote_handle_open_fn = decltype(remote_handle_open);
using remote_handle64_open_fn = decltype(remote_handle64_open);
using remote_handle_invoke_fn = decltype(remote_handle_invoke);
using remote_handle64_invoke_fn = decltype(remote_handle64_invoke);
using remote_handle_close_fn = decltype(remote_handle_close);
using remote_handle64_close_fn = decltype(remote_handle64_close);
using remote_mmap_fn = decltype(remote_mmap);
using remote_mmap64_fn = decltype(remote_mmap64);
using remote_munmap_fn = decltype(remote_munmap);
using remote_munmap64_fn = decltype(remote_munmap64);
using remote_register_buf_fn = decltype(remote_register_buf);
using remote_set_mode_fn = decltype(remote_set_mode);
using remote_handle_control_fn = decltype(remote_handle_control);

struct AdsprpcInterface {
  remote_handle_open_fn* handle_open_fn =
      reinterpret_cast<remote_handle_open_fn*>(
          LoadFunction("remote_handle_open"));
  remote_handle64_open_fn* handle64_open_fn =
      reinterpret_cast<remote_handle64_open_fn*>(
          LoadFunction("remote_handle64_open"));
  remote_handle_invoke_fn* handle_invoke_fn =
      reinterpret_cast<remote_handle_invoke_fn*>(
          LoadFunction("remote_handle_invoke"));
  remote_handle64_invoke_fn* handle64_invoke_fn =
      reinterpret_cast<remote_handle64_invoke_fn*>(
          LoadFunction("remote_handle64_invoke"));
  remote_handle_close_fn* handle_close_fn =
      reinterpret_cast<remote_handle_close_fn*>(
          LoadFunction("remote_handle_close"));
  remote_handle64_close_fn* handle64_close_fn =
      reinterpret_cast<remote_handle64_close_fn*>(
          LoadFunction("remote_handle64_close"));
  remote_handle_control_fn* handle_control_fn =
      reinterpret_cast<remote_handle_control_fn*>(
          LoadFunction("remote_handle_control"));
  remote_mmap_fn* mmap_fn =
      reinterpret_cast<remote_mmap_fn*>(LoadFunction("remote_mmap"));
  remote_munmap_fn* munmap_fn =
      reinterpret_cast<remote_munmap_fn*>(LoadFunction("remote_munmap"));
  remote_mmap64_fn* mmap64_fn =
      reinterpret_cast<remote_mmap64_fn*>(LoadFunction("remote_mmap64"));
  remote_munmap64_fn* munmap64_fn =
      reinterpret_cast<remote_munmap64_fn*>(LoadFunction("remote_munmap64"));
  remote_register_buf_fn* register_buf_fn =
      reinterpret_cast<remote_register_buf_fn*>(
          LoadFunction("remote_register_buf"));
  remote_set_mode_fn* set_mode_fn =
      reinterpret_cast<remote_set_mode_fn*>(LoadFunction("remote_set_mode"));

  // Returns singleton instance.
  static AdsprpcInterface* Singleton() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_4(mht_4_v, 305, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "Singleton");

    static AdsprpcInterface* instance = new AdsprpcInterface();
    return instance;
  }
};

}  // namespace

extern "C" {
int remote_handle_open(const char* name, remote_handle* h) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_5(mht_5_v, 318, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_handle_open");

  return AdsprpcInterface::Singleton()->handle_open_fn
             ? AdsprpcInterface::Singleton()->handle_open_fn(name, h)
             : -1;
}

int remote_handle64_open(const char* name, remote_handle64* h) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_6(mht_6_v, 328, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_handle64_open");

  return AdsprpcInterface::Singleton()->handle64_open_fn
             ? AdsprpcInterface::Singleton()->handle64_open_fn(name, h)
             : -1;
}

int remote_handle_invoke(remote_handle h, uint32_t scalars, remote_arg* args) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_7(mht_7_v, 337, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_handle_invoke");

  return AdsprpcInterface::Singleton()->handle_invoke_fn
             ? AdsprpcInterface::Singleton()->handle_invoke_fn(h, scalars, args)
             : -1;
}

int remote_handle64_invoke(remote_handle64 h, uint32_t scalars,
                           remote_arg* args) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_8(mht_8_v, 347, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_handle64_invoke");

  return AdsprpcInterface::Singleton()->handle64_invoke_fn
             ? AdsprpcInterface::Singleton()->handle64_invoke_fn(h, scalars,
                                                                 args)
             : -1;
}

int remote_handle_close(remote_handle h) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_9(mht_9_v, 357, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_handle_close");

  return AdsprpcInterface::Singleton()->handle_close_fn
             ? AdsprpcInterface::Singleton()->handle_close_fn(h)
             : -1;
}

int remote_handle64_close(remote_handle64 h) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_10(mht_10_v, 366, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_handle64_close");

  return AdsprpcInterface::Singleton()->handle64_close_fn
             ? AdsprpcInterface::Singleton()->handle64_close_fn(h)
             : -1;
}

int remote_handle_control(uint32_t req, void* data, uint32_t datalen) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_11(mht_11_v, 375, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_handle_control");

  return AdsprpcInterface::Singleton()->handle_control_fn
             ? AdsprpcInterface::Singleton()->handle_control_fn(req, data,
                                                                datalen)
             : -1;
}

int remote_mmap(int fd, uint32_t flags, uint32_t addr, int size,
                uint32_t* result) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_12(mht_12_v, 386, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_mmap");

  return AdsprpcInterface::Singleton()->mmap_fn
             ? AdsprpcInterface::Singleton()->mmap_fn(fd, flags, addr, size,
                                                      result)
             : -1;
}

int remote_mmap64(int fd, uint32_t flags, uintptr_t vaddrin, int64_t size,
                  uintptr_t* vaddrout) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_13(mht_13_v, 397, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_mmap64");

  return AdsprpcInterface::Singleton()->mmap64_fn
             ? AdsprpcInterface::Singleton()->mmap64_fn(fd, flags, vaddrin,
                                                        size, vaddrout)
             : -1;
}

int remote_munmap(uint32_t addr, int size) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_14(mht_14_v, 407, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_munmap");

  return AdsprpcInterface::Singleton()->munmap_fn
             ? AdsprpcInterface::Singleton()->munmap_fn(addr, size)
             : -1;
}

int remote_munmap64(uintptr_t vaddrout, int64_t size) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_15(mht_15_v, 416, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_munmap64");

  return AdsprpcInterface::Singleton()->munmap64_fn
             ? AdsprpcInterface::Singleton()->munmap64_fn(vaddrout, size)
             : -1;
}

void remote_register_buf(void* buf, int size, int fd) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_16(mht_16_v, 425, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_register_buf");

  if (AdsprpcInterface::Singleton()->register_buf_fn) {
    AdsprpcInterface::Singleton()->register_buf_fn(buf, size, fd);
  }
}

int remote_set_mode(uint32_t mode) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPShexagon_nnPSadsprpc_interfaceDTcc mht_17(mht_17_v, 434, "", "./tensorflow/lite/delegates/hexagon/hexagon_nn/adsprpc_interface.cc", "remote_set_mode");

  return AdsprpcInterface::Singleton()->set_mode_fn
             ? AdsprpcInterface::Singleton()->set_mode_fn(mode)
             : -1;
}

}  // extern "C"
