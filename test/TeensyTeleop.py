# THE FOLLOWING CODE RUNS A TELEOP SERVER BETWEEN THE PI AND TEENSY.
# REQUIRES MOTOR CODE UPLOADED TO TEENSY AND TELEOP SCRIPT ON LAPTOP.

import serial
import socket
import glob
import time

UDP_PORT = 5005
BAUD = 115200

def find_teensy():
    ports = glob.glob("/dev/ttyACM*")
    if not ports:
        raise Exception("No Teensy found!")
    return ports[0]

def main():
    teensy_port = find_teensy()
    ser = serial.Serial(teensy_port, BAUD, timeout=0.1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))

    print("Teleop server running, forwarding UDP to Teensy")

    while True:
        data, addr = sock.recvfrom(1024)
        cmd = data.decode().strip()
        if cmd:
            print("Forwarding command to Teensy:", cmd)
            ser.write((cmd + "\n").encode())

        # Read Teensy feedback
        if ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            print("Teensy feedback:", line)

if __name__ == "__main__":
    time.sleep(2)  # let serial initialize
    main()
				
