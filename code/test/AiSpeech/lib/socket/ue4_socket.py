# -*- coding: utf-8 -*-
# !/usr/bin/env python36
"""
    tgshg/socket/ue4_socket_format.py
    :model: UE4 Socket Format
    :copyright:facegood © 2019 by the tang.
"""
import os
import sys
import numpy as np

import threading
import socket
from contextlib import contextmanager
import time
BUFF_SIZE = 1024
RECORDING = False
RECORDING_BEGIN = "1"
RECORDING_END = "2"

# Thread-local state to stored information on locks already acquired
_local = threading.local()

@contextmanager
def acquire(*locks):
    # Sort locks by object identifier
    locks = sorted(locks, key=lambda x: id(x))

    # Make sure lock order of previously acquired locks is not violated
    acquired = getattr(_local,'acquired',[])
    if acquired and max(id(lock) for lock in acquired) >= id(locks[0]):
        raise RuntimeError('Lock Order Violation')

    # Acquire all of the locks
    acquired.extend(locks)
    _local.acquired = acquired

    try:
        for lock in locks:
            lock.acquire()
        yield
    finally:
        # Release locks in reverse order of acquisition
        for lock in reversed(locks):
            lock.release()
        del acquired[-len(locks):]

x_lock = threading.Lock()
y_lock = threading.Lock()

class UdpRecvHandler:
    def __init__(self, addr_bind):
        self.addr_bind = addr_bind
        self.udp =  socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setblocking(1)
        self.udp.bind(self.addr_bind)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        thread = threading.Thread(None,target = self.recv_handler)
        thread.start()

    def recv_handler(self):
        global RECORDING
        while True:
            time.sleep(0.01)
            with acquire(x_lock, y_lock):
                msg, addr = self.udp.recvfrom(BUFF_SIZE)
            # print("msg:",msg)
            len_recv = int(np.frombuffer(msg[:4],dtype='<u4'))
            if msg[-1:] == b'\x00':
                recv_msg = msg[4:len_recv+3]
                # print("recv_msg:",recv_msg)
                # print("addr:",addr)
                if recv_msg == RECORDING_BEGIN:
                    with acquire(x_lock, y_lock):
                        RECORDING = True
                elif recv_msg == RECORDING_END:
                    with acquire(x_lock, y_lock):
                        RECORDING = False
                else:
                    print("Unknown",recv_msg)


class UdpSendHandler:
    def __init__(self, addr_send):
        self.addr_send = addr_send
        self.udp =  socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.setblocking(1)
        self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send_handler(self,data):
        data = np.array(data, dtype='float32')
        data_char = data.tobytes()
        send_data = data_char + b'\x00\x00\x00\x00'
        self.udp.sendto(send_data, self.addr_send)