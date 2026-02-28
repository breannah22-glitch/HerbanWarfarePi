## THE FOLLOWING CODE RECEIVES UDP FROM LAPTOP AND SENDS COMMANDS TO
## AN ARDUINO UNO R3. PI RECEIVES CONFIRMATION FROM ARDUINO AS WELL

import serial
import socket
import glob
import time

UDP_PORT = 5005
BAUD = 115200

def find_arduino():
    ports = glob.glob("/dev/ttyACM*")
    if not ports:
        raise Exception("No Arduino found!")
    return ports[0]

def main():
    teensy_port = find_arduino()
    ser = serial.Serial(teensy_port, BAUD, timeout=0.1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))

    print("Teleop server running, forwarding UDP to Arduino")

    while True:
        data, addr = sock.recvfrom(1024)
        cmd = data.decode().strip()
        if cmd:
            print("Forwarding command to Arduino:", cmd)
            ser.write((cmd + "\n").encode())

        # Optional: read Arduino feedback
        if ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            print("Arduino feedback:", line)

if __name__ == "__main__":
    time.sleep(2)  # let serial initialize
    main()
