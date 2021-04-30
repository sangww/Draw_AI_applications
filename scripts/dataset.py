from torch.utils.data import Dataset, DataLoader
from scipy import signal
import numpy as np

def load_strokes(file_path):  
    strokes = []
    with open(file_path) as f:
        start = False
        stroke = []
        for line in f.readlines():
            if line and line.startswith("stroke"):
                start = True
                stroke = []
                continue
            if start:
                if len(line) == 1:
                    start = False # reach the end of stroke
                    strokes.append(stroke.copy())
                    stroke = []
                elif len(line) > 30:
                    stroke.append(line)
        return strokes
        
def convert_strokes_to_points(strokes):
    processed_strokes = []
    for stroke in strokes:
        points = []
        for point in stroke:
            points.append([float(i) for i in point.split()])
        #print("length = ", len(points))
        processed_strokes.append(points.copy())
    return processed_strokes
    
# output format
# ready to use in neural networks
# 0 delta x
# 1 delta y
# 2 pressure
# 3 tilt X
# 4 tilt Y
# 5 x0
# 6 y0

delta_scale = 100

def normalize_strokes(processed_strokes, sequence_length):
    normalized = []
    max_length = 0
    for stroke in processed_strokes:
        if len(stroke) > max_length:
            max_length = len(stroke)
        if len(stroke) > sequence_length:
            # need to resample to the max length
            n = len(stroke)

            i = 0
            x0 = stroke[0][2]
            y0 = stroke[0][3]
            lastx = x0
            lasty = y0
            while n > 0:
                points = []
                for p in stroke[i:min(i+sequence_length, len(stroke))]:
                  # remove rotation
                    points.append([(p[2] - lastx)*delta_scale, (p[3] - lasty)*delta_scale, p[4], p[6], p[7], x0, y0])
                    lastx, lasty = p[2], p[3]
                    
                for j in range(sequence_length - len(points)):
                    points.append([0, 0, 0, 0, 0, 0, 0])
                n -= sequence_length
                normalized.append(points.copy())
            continue 
        points = []
        x0 = stroke[0][2]
        y0 = stroke[0][3]
        lastx = x0
        lasty = y0
        for p in stroke:
          # remove rotation
            points.append([(p[2] - lastx)*delta_scale, (p[3] - lasty)*delta_scale, p[4], p[6], p[7], x0, y0])
            lastx, lasty = p[2], p[3]
            
        for j in range(sequence_length - len(points)):
            points.append([0, 0, 0, 0, 0, 0, 0])
        normalized.append(points.copy())
    print("max_length=",max_length)
    return normalized
    
def rebuild_strokes(normalized_strokes, delta_t):
    result = []
    for stroke in normalized_strokes:
        x0, y0 = stroke[0][5], stroke[0][6]
        lastx, lasty = x0, y0
        index = 0
        time = 0.0
        points = []
        for point in stroke:
            new_point = [index, time, lastx + point[0], lasty + point[1], point[2], 0.0, point[3], point[4], point[5]]
            points.append(new_point)
            time += delta_t
            index += 1
            lastx, lasty = new_point[2], new_point[3]
        result.append(points.copy())
    return result

class CynData(Dataset):
    def __init__(self, filename, sequence_length):
        strokes = load_strokes(filename)
        stroke_points = convert_strokes_to_points(strokes)
        self.training_data = np.transpose(np.array(normalize_strokes(stroke_points, sequence_length)), (0, 2, 1))[:, :5, :]
        self.training_data = self.training_data.astype(np.float32)
        
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        # dimension 0: point index in a stroke
        # dimension 1: pen stroke dim ( = 6)
        return self.training_data[idx]
        