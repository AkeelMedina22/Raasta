import json
rs = 0
samples = 0
normal_road = 0
bad_road = 0
speedbreaker = 0
pothole = 0
with open('raasta-c542d-default-rtdb-export.json', 'r') as f:
  data = json.load(f)
  for x in data["sensor-data"]:
    rs += 1
    for y in data["sensor-data"][x]:
        samples += 1
        for z in data["sensor-data"][x][y]:
            if data["sensor-data"][x][y][z] == "Normal Road":
                normal_road += 1
            elif data["sensor-data"][x][y][z] == "Speedbreaker":
                speedbreaker += 1
            elif data["sensor-data"][x][y][z] == "Bad Road":
                bad_road += 1
            elif data["sensor-data"][x][y][z] == "Pothole":
                pothole += 1

print(rs)
print(samples)
print(normal_road)
print(pothole)
print(speedbreaker)
print(bad_road)

