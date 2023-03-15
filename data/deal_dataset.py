import splitfolders


splitfolders.ratio(input='../runs/input', output='m3fd-detect', seed=1337, ratio=(0.8, 0.1, 0.1))
splitfolders.ratio(input='../runs/input_label', output='m3fd-detect-label', seed=1337, ratio=(0.8, 0.1, 0.1))
