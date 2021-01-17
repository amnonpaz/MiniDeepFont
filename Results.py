from numpy import zeros, argmax
import csv
import Fonts

def store(dest_filename, predictions, filenames, letters):
    with open(dest_filename, 'w', newline='') as csvfile:
        reswriter = csv.writer(csvfile, delimiter=',')
        reswriter.writerow(['', 'image','char'] + Fonts.get_list())
        idx = 0
        for v in argmax(predictions, axis=1):
            res = zeros(Fonts.get_number())
            res[v] = 1.0
            reswriter.writerow([idx] + [filenames[idx], letters[idx]] + res.tolist())
            idx += 1
