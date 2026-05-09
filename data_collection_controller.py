import subprocess
import time
import os

log_file = "collection_output.log"
with open(log_file, "w") as f:
    f.write("Collection started\n")

p = subprocess.Popen(
    ["python", "-u", "serial_collector.py", "--port", "COM10"],
    stdin=subprocess.PIPE,
    stdout=open(log_file, "a"),
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

time.sleep(5)

# Use the new "start 2" command we added
print("Sending: start 2")
p.stdin.write("start 2\n")
p.stdin.flush()

try:
    while p.poll() is None:
        time.sleep(10)
except KeyboardInterrupt:
    p.stdin.write("stop\n")
    p.stdin.write("Q\n")
    p.stdin.flush()
    p.terminate()
