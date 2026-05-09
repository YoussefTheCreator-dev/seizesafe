import socket
import time
import pandas as pd
import os

# --- CONFIG ---
HOST = '127.0.0.1'
PORT = 8888
DATA_FILE = 'dataset/S01R01.txt' 

def start_simulator():
    file_to_use = DATA_FILE if os.path.exists(DATA_FILE) else 'dataset_fog_release/dataset/S01R01.txt'
    if not os.path.exists(file_to_use):
        print("❌ Error: Daphnet dataset files not found.")
        return

    print(f"📖 Replaying Ankle-only data: {file_to_use}...")
    cols = ['time', 'ax', 'ay', 'az', 'lx', 'ly', 'lz', 'tx', 'ty', 'tz', 'label']
    df = pd.read_csv(file_to_use, sep=' ', header=None, names=cols)
    df = df[df['label'] > 0]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"📡 Ankle Streamer (Watch Simulator) listening on {HOST}:{PORT}...")
        
        conn, addr = s.accept()
        with conn:
            print(f"✅ Dashboard connected!")
            conn.recv(1024) 
            
            for _, row in df.iterrows():
                # Stream ONLY Ankle: ax, ay, az
                line = f"{int(row['time'])},{row['ax']},{row['ay']},{row['az']},{int(row['label'])}\n"
                conn.sendall(line.encode())
                time.sleep(0.02) # 50Hz
                
if __name__ == "__main__":
    start_simulator()
