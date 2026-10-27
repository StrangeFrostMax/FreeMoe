/*
 * vnpu-node - standalone executable form of the virtual NPU.
 */
#include "vnpu.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
void stop_on_signal(int) { vnpu_node_stop(); }
}

int main(int argc, char **argv)
{
    vnpu_node_config_t cfg;
    vnpu_node_config_init(&cfg);
    std::string bind = "0.0.0.0";
    std::string token;
    std::string name = "vnpu-node";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };
        if (a == "--port") cfg.port = static_cast<uint16_t>(std::atoi(next()));
        else if (a == "--threads") cfg.num_threads = std::atoi(next());
        else if (a == "--bind") bind = next();
        else if (a == "--token") token = next();
        else if (a == "--name") name = next();
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: vnpu-node [--port N] [--threads N] "
                        "[--bind address] [--token secret] [--name name]\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return 2;
        }
    }
    cfg.bind_address = bind.c_str();
    cfg.auth_token = token.empty() ? nullptr : token.c_str();
    cfg.node_name = name.c_str();
    std::signal(SIGINT, stop_on_signal);
    std::signal(SIGTERM, stop_on_signal);
    vnpu_status_t rc = vnpu_node_run(&cfg);
    if (rc != VNPU_OK) {
        std::fprintf(stderr, "vnpu-node failed: %s\n",
                     vnpu_status_string(rc));
        return 1;
    }
    return 0;
}
