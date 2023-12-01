# How to use
The detected data must be yolo liked format where each image is associated a .txt file like
![Selection_171](https://github.com/george-free/cv_MAP/assets/29144898/ad89e62a-226e-4765-83cc-da4ff267dc23)

Each txt saved the detected results like
```
class_name score x1 y1 x2 y2
...
```
![Selection_172](https://github.com/george-free/cv_MAP/assets/29144898/d41c70de-0d0a-4b54-9a8c-aeec649bf8ff)

The ground truth has the format similiar detected results.

![Selection_189](https://github.com/george-free/cv_MAP/assets/29144898/f913b00b-78b2-4656-b17c-47702aee4cb3)


run the script below

```
python3 voc_map.py -d dir/to/detected_results -g dir/to/groundtruth
```
