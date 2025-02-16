/*
 *
 * Copyright 2021-2024 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#include "scheduler_metrics_sender.h"
#include "fmt/format.h"
#include <mutex>
#include <iostream>
#include <fcntl.h>      // For fcntl flags
#include <sys/socket.h> // For socket functions
#include <netinet/in.h> // For sockaddr_in
#include <unistd.h>     // For close()

using namespace srsran;

scheduler_metrics_sender::scheduler_metrics_sender() : 
    server_fd(-1), 
    client_fd(-1), 
    running(false),
    logger(srslog::fetch_basic_logger("SCHED")) {}

bool scheduler_metrics_sender::init(int port)
{
    std::lock_guard<std::mutex> lock(mutex);
    
    // Check if already running
    if (running) {
        return true;
    }

    int opt = 1;
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        logger.error("Failed to create socket: {}", strerror(errno));
        return false;
    }

    // Set both SO_REUSEADDR and SO_REUSEPORT to allow socket reuse
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        logger.error("Failed to set socket options: {}", strerror(errno));
        close(server_fd);
        return false;
    }

    struct sockaddr_in server_addr = {};  // Zero initialize
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    // Try binding with detailed error reporting
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        logger.error("Failed to bind to port {}: {} (errno={})", port, strerror(errno), errno);
        close(server_fd);
        return false;
    }

    if (listen(server_fd, 1) < 0) {
        logger.error("Failed to listen on socket: {}", strerror(errno));
        close(server_fd);
        return false;
    }

    // Set non-blocking mode after successful bind
    int flags = fcntl(server_fd, F_GETFL, 0);
    fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);

    logger.info("Metrics server successfully listening on port {}", port);
    running = true;
    return true;
}

void scheduler_metrics_sender::stop()
{
    std::lock_guard<std::mutex> lock(mutex);
    running = false;
    
    // Close client socket if open
    if (client_fd >= 0) {
        close(client_fd);
        client_fd = -1;
    }
    
    // Close server socket if open
    if (server_fd >= 0) {
        close(server_fd);
        server_fd = -1;
    }
}

bool scheduler_metrics_sender::send_message(const std::string& msg)
{
    std::lock_guard<std::mutex> lock(mutex);
    
    if (!running) {
        if (!init(DEFAULT_METRICS_PORT)) {
            return false;
        }
    }

    // Non-blocking check for new connections
    if (client_fd < 0) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int new_client = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        
        if (new_client >= 0) {
            // Set non-blocking for client socket
            int flags = fcntl(new_client, F_GETFL, 0);
            fcntl(new_client, F_SETFL, flags | O_NONBLOCK);
            client_fd = new_client;
        }
        return false;  // Skip sending if no client
    }

    // Non-blocking send
    ssize_t sent = send(client_fd, msg.c_str(), msg.length(), MSG_DONTWAIT | MSG_NOSIGNAL);
    if (sent < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            // Just close client socket and wait for new connection
            close(client_fd);
            client_fd = -1;
            // Don't reinit server socket, just wait for new client
            return false;
        }
        return false;
    }
    return true;
}

std::string scheduler_metrics_sender::format_prb_metrics(const prb_metrics& metrics)
{
    return fmt::format("TYPE=PRB,UE_IDX={},RNTI={:#6x},PRBs={},SLOT={}\n", 
                      metrics.ue_index,
                      (unsigned)metrics.crnti,
                      metrics.nof_prbs,
                      metrics.slot.to_uint());
}

std::string scheduler_metrics_sender::format_sr_metrics(const sr_metrics& metrics)
{
    return fmt::format("TYPE=SR,UE_IDX={},RNTI={:#6x},SLOT={}\n",
                      metrics.ue_index,
                      (unsigned)metrics.crnti,
                      metrics.slot.to_uint());
}

bool scheduler_metrics_sender::send_prb_metrics(const prb_metrics& metrics)
{
    std::string msg = format_prb_metrics(metrics);
    return send_message(msg);
}

bool scheduler_metrics_sender::send_sr_metrics(const sr_metrics& metrics)
{
    std::string msg = format_sr_metrics(metrics);
    return send_message(msg);
}

std::string scheduler_metrics_sender::format_bsr_metrics(const bsr_metrics& metrics)
{
    return fmt::format("TYPE=BSR,UE_IDX={},RNTI={:#6x},BYTES={},SLOT={}\n",
                      metrics.ue_index,
                      (unsigned)metrics.crnti,
                      metrics.nof_bytes,
                      metrics.slot_rx.to_uint());
}

bool scheduler_metrics_sender::send_bsr_metrics(const bsr_metrics& metrics)
{
    std::string msg = format_bsr_metrics(metrics);
    return send_message(msg);
} 