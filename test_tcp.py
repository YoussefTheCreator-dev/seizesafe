import socket
import time

def test_connection():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 8888))
        print("Connected to server!")
        for i in range(20):
            # Send fake IMU line: timestamp,ax,ay,az,gx,gy,gz
            data = f"{int(time.time()*1000)},0.1,0.2,0.9,0.01,0.02,0.03\n"
            s.send(data.encode())
            time.sleep(0.1)
        s.close()
        print("Test data sent.")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_connection()
