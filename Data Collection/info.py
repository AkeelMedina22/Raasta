import json
rs = 0
samples = 0
normal_road = 0
bad_road = 0
speedbreaker = 0
pothole = 0
sudden_change = 0
traffic = 0

with open('raasta-c542d-default-rtdb-sensor-data2-export.json', 'r') as f:
    data = json.load(f)
    for key in data:
        rs += 1
        for timestamp in data[key]:
            samples += 1
            for l in data[key][timestamp]:
                if data[key][timestamp][l] == "Normal Road":
                    normal_road += 1
                elif data[key][timestamp][l] == "Bad Road":
                    bad_road += 1
                elif data[key][timestamp][l] == "Speedbreaker":
                    speedbreaker += 1
                elif data[key][timestamp][l] == "Pothole":
                    pothole += 1
                elif data[key][timestamp][l] == "Sudden Change":
                    sudden_change += 1
                elif data[key][timestamp][l] == "Traffic":
                    traffic += 1

print("TOTAL ROAD SEGMENTS COLLECTED: ", rs)
print("TOTAL ROAD SAMPLES COLLECTED: ", samples)
print("NORMAL ROAD SAMPLES: ", normal_road)
print("POTHOLE ROAD SAMPLES: ", pothole)
print("SPEEDBREAKER ROAD SAMPLES: ", speedbreaker)
print("BAD ROAD SAMPLES: ", bad_road)
print("SUDDEN CHANGE SAMPLES: ", sudden_change)
print("TRAFFIC SAMPLES: ", traffic)

