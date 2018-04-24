from __future__ import division
import threading
import numpy as np
import serial
import time

DEVICE_SYNC = 0X3A
DEVICE_ADDRESS = 0X01
DEVICE_ZOOM = 0X00
DEVICE_FUNCTION = 0X10

class monitor_driver():
    def __init__(self, shape, object_box, gain):
        self.im_height = shape[1] # PIL image.size (w, h) cv image.shape (h, w, c)
        self.im_width = shape[0]
        self.im_centre = [self.im_width / 2, self.im_height / 2] # (x, y) image centre
        self.object = object_box
        self.object_centre = [self.object[0] + self.object[2] / 2, self.object[1] + self.object[3] / 2]
        self.serial_data = range(9)
        self.gain = gain
        self.lock = threading._allocate_lock()

    def compute_parity(self, data):
        bytes_sum = sum(data) - data[0] - data[8]
        return bytes_sum % 256

    # dec -> hex -> high&low byte
    def serial_format(self, data):
        hex_str = hex(int(data) & 0XFFFF)
        if len(hex_str) == 6:
            high_byte = hex_str[2:4]
            low_byte = hex_str[4:6]
            high_int = int(high_byte, 16)
            low_int = int(low_byte, 16)
        elif len(hex_str) == 5:
            high_byte = hex_str[2]
            low_byte = hex_str[3:5]
            high_int = int(high_byte, 16)
            low_int = int(low_byte, 16)
        elif len(hex_str) == 4:
            low_byte = hex_str[2:4]
            high_int = 0
            low_int = int(low_byte, 16)
        else:
            low_byte = hex_str[2]
            high_int = 0
            low_int = int(low_byte, 16)
        return high_int, low_int


    def tracking(self):
        self.lock.acquire()
        try:
            hori_coord = - self.gain * ((self.object_centre[0] - self.im_centre[0]) / self.im_centre[0]) * 8192
            vert_coord = self.gain * ((self.object_centre[1] - self.im_centre[1]) / self.im_centre[1]) * 8192

            self.serial_data[0] = int(DEVICE_SYNC)
            self.serial_data[1] = int(DEVICE_ADDRESS)
            self.serial_data[2] = int(DEVICE_FUNCTION)

            self.serial_data[3] = self.serial_format(hori_coord)[0]
            self.serial_data[4] = self.serial_format(hori_coord)[1]
            self.serial_data[5] = self.serial_format(vert_coord)[0]
            self.serial_data[6] = self.serial_format(vert_coord)[1]

            self.serial_data[7] = int(DEVICE_ZOOM)
            self.serial_data[8] = self.compute_parity(self.serial_data)

        finally:
            self.lock.release()

        return self.serial_data




