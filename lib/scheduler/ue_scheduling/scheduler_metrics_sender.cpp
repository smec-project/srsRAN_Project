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
#include <arpa/inet.h> // For inet_addr
#include <fcntl.h>     // For fcntl flags
#include <iostream>
#include <mutex>
#include <netinet/in.h> // For sockaddr_in
#include <sys/socket.h> // For socket functions
#include <unistd.h>     // For close()

using namespace srsran;

scheduler_metrics_sender::scheduler_metrics_sender() :
  server_fd(-1), running(false), logger(srslog::fetch_basic_logger("SCHED"))
{
}

bool scheduler_metrics_sender::init(int port)
{
  std::lock_guard<std::mutex> lock(mutex);

  // Check if already running
  if (running) {
    return true;
  }

  server_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (server_fd < 0) {
    logger.error("Failed to create socket: {}", strerror(errno));
    return false;
  }

  // Set non-blocking mode
  int flags = fcntl(server_fd, F_GETFL, 0);
  fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);

  logger.info("Metrics UDP sender initialized (sending to port {})", port);
  running = true;
  return true;
}

void scheduler_metrics_sender::stop()
{
  std::lock_guard<std::mutex> lock(mutex);
  running = false;

  // Close server socket if open
  if (server_fd >= 0) {
    close(server_fd);
    server_fd = -1;
  }
}

bool scheduler_metrics_sender::send_binary_message(const metrics_message& msg)
{
  std::lock_guard<std::mutex> lock(mutex);

  if (!running) {
    if (!init(DEFAULT_METRICS_PORT)) {
      return false;
    }
  }

  // UDP send to localhost (127.0.0.1)
  struct sockaddr_in client_addr = {};
  client_addr.sin_family         = AF_INET;
  client_addr.sin_addr.s_addr    = inet_addr("127.0.0.1");
  client_addr.sin_port           = htons(DEFAULT_METRICS_PORT);

  // Non-blocking send binary data via UDP
  ssize_t sent = sendto(
      server_fd, &msg, sizeof(msg), MSG_DONTWAIT | MSG_NOSIGNAL, (struct sockaddr*)&client_addr, sizeof(client_addr));
  if (sent < 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      logger.error("Failed to send metrics via UDP: {}", strerror(errno));
      return false;
    }
    return false;
  }
  return true;
}

scheduler_metrics_sender::metrics_message scheduler_metrics_sender::create_prb_message(const prb_metrics& metrics)
{
  // Format: [type=0][rnti][prbs][slot]
  metrics_message msg;
  msg.type   = 0;
  msg.rnti   = (uint32_t)metrics.crnti;
  msg.field1 = metrics.nof_prbs;
  msg.field2 = metrics.slot.to_uint();
  return msg;
}

scheduler_metrics_sender::metrics_message scheduler_metrics_sender::create_sr_message(const sr_metrics& metrics)
{
  // Format: [type=1][rnti][slot][0]
  metrics_message msg;
  msg.type   = 1;
  msg.rnti   = (uint32_t)metrics.crnti;
  msg.field1 = metrics.slot.to_uint();
  msg.field2 = 0; // Unused field for SR
  return msg;
}

bool scheduler_metrics_sender::send_prb_metrics(const prb_metrics& metrics)
{
  metrics_message msg = create_prb_message(metrics);
  return send_binary_message(msg);
}

bool scheduler_metrics_sender::send_sr_metrics(const sr_metrics& metrics)
{
  metrics_message msg = create_sr_message(metrics);
  return send_binary_message(msg);
}

scheduler_metrics_sender::metrics_message scheduler_metrics_sender::create_bsr_message(const bsr_metrics& metrics)
{
  // Format: [type=2][rnti][bytes][slot]
  metrics_message msg;
  msg.type   = 2;
  msg.rnti   = (uint32_t)metrics.crnti;
  msg.field1 = metrics.nof_bytes;
  msg.field2 = metrics.slot_rx.to_uint();
  return msg;
}

bool scheduler_metrics_sender::send_bsr_metrics(const bsr_metrics& metrics)
{
  metrics_message msg = create_bsr_message(metrics);
  return send_binary_message(msg);
}