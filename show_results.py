import matplotlib.pyplot as plt
import numpy as np

# Extract data
train_loss = []
eval_loss = []
accuracy = []
epochs = []



data = '''
[INFO]: Epoch 1 of 100
Training ist starting...
Average Loss: 0.554945
Evaluating...
Average Loss: 0.408908
Final accuracy: 81.1560%
[INFO]: Epoch 2 of 100
Training ist starting...
Average Loss: 0.386196
Evaluating...
Average Loss: 0.350748
Final accuracy: 84.6800%
[INFO]: Epoch 3 of 100
Training ist starting...
Average Loss: 0.356623
Evaluating...
Average Loss: 0.383983
Final accuracy: 82.3960%
[INFO]: Epoch 4 of 100
Training ist starting...
Average Loss: 0.341578
Evaluating...
Average Loss: 0.338415
Final accuracy: 85.3800%
[INFO]: Epoch 5 of 100
Training ist starting...
Average Loss: 0.345948
Evaluating...
Average Loss: 0.350361
Final accuracy: 84.6840%
[INFO]: Epoch 6 of 100
Training ist starting...
Average Loss: 0.351674
Evaluating...
Average Loss: 0.371008
Final accuracy: 83.6560%
[INFO]: Epoch 7 of 100
Training ist starting...
Average Loss: 0.337455
Evaluating...
Average Loss: 0.334685
Final accuracy: 85.4640%
[INFO]: Epoch 8 of 100
Training ist starting...
Average Loss: 0.330943
Evaluating...
Average Loss: 0.366455
Final accuracy: 83.9600%
[INFO]: Epoch 9 of 100
Training ist starting...
Average Loss: 0.320231
Evaluating...
Average Loss: 0.349494
Final accuracy: 84.8560%
[INFO]: Epoch 10 of 100
Training ist starting...
Average Loss: 0.306825
Evaluating...
Average Loss: 0.333939
Final accuracy: 85.3960%
[INFO]: Epoch 11 of 100
Training ist starting...
Average Loss: 0.298180
Evaluating...
Average Loss: 0.332538
Final accuracy: 85.6960%
[INFO]: Epoch 12 of 100
Training ist starting...
Average Loss: 0.292069
Evaluating...
Average Loss: 0.343181
Final accuracy: 85.4400%
[INFO]: Epoch 13 of 100
Training ist starting...
Average Loss: 0.285714
Evaluating...
Average Loss: 0.328761
Final accuracy: 85.9120%
[INFO]: Epoch 14 of 100
Training ist starting...
Average Loss: 0.279316
Evaluating...
Average Loss: 0.339641
Final accuracy: 85.4760%
[INFO]: Epoch 15 of 100
Training ist starting...
Average Loss: 0.281411
Evaluating...
Average Loss: 0.331512
Final accuracy: 85.6840%
[INFO]: Epoch 16 of 100
Training ist starting...
Average Loss: 0.275674
Evaluating...
Average Loss: 0.337883
Final accuracy: 85.7560%
[INFO]: Epoch 17 of 100
Training ist starting...
Average Loss: 0.274273
Evaluating...
Average Loss: 0.348766
Final accuracy: 85.3280%
[INFO]: Epoch 18 of 100
Training ist starting...
Average Loss: 0.264867
Evaluating...
Average Loss: 0.335062
Final accuracy: 85.7400%
[INFO]: Epoch 19 of 100
Training ist starting...
Average Loss: 0.247964
Evaluating...
Average Loss: 0.340044
Final accuracy: 85.8360%
[INFO]: Epoch 20 of 100
Training ist starting...
Average Loss: 0.238590
Evaluating...
Average Loss: 0.333700
Final accuracy: 86.0720%
[INFO]: Epoch 21 of 100
Training ist starting...
Average Loss: 0.234486
Evaluating...
Average Loss: 0.335131
Final accuracy: 85.8880%
[INFO]: Epoch 22 of 100
Training ist starting...
Average Loss: 0.228062
Evaluating...
Average Loss: 0.344838
Final accuracy: 85.9480%
[INFO]: Epoch 23 of 100
Training ist starting...
Average Loss: 0.215363
Evaluating...
Average Loss: 0.348252
Final accuracy: 85.7880%
[INFO]: Epoch 24 of 100
Training ist starting...
Average Loss: 0.205347
Evaluating...
Average Loss: 0.351894
Final accuracy: 85.6480%
[INFO]: Epoch 25 of 100
Training ist starting...
Average Loss: 0.193095
Evaluating...
Average Loss: 0.354349
Final accuracy: 85.8320%
[INFO]: Epoch 26 of 100
Training ist starting...
Average Loss: 0.186834
Evaluating...
Average Loss: 0.368687
Final accuracy: 85.5720%
[INFO]: Epoch 27 of 100
Training ist starting...
Average Loss: 0.178794
Evaluating...
Average Loss: 0.362699
Final accuracy: 85.7720%
[INFO]: Epoch 28 of 100
Training ist starting...
Average Loss: 0.172427
Evaluating...
Average Loss: 0.370422
Final accuracy: 85.8520%
[INFO]: Epoch 29 of 100
Training ist starting...
Average Loss: 0.162826
Evaluating...
Average Loss: 0.370575
Final accuracy: 85.7640%
[INFO]: Epoch 30 of 100
Training ist starting...
Average Loss: 0.154427
Evaluating...
Average Loss: 0.364347
Final accuracy: 85.9440%
[INFO]: Epoch 31 of 100
Training ist starting...
Average Loss: 0.148932
Evaluating...
Average Loss: 0.390616
Final accuracy: 85.5480%
[INFO]: Epoch 32 of 100
Training ist starting...
Average Loss: 0.138162
Evaluating...
Average Loss: 0.400958
Final accuracy: 85.6080%
[INFO]: Epoch 33 of 100
Training ist starting...
Average Loss: 0.133718
Evaluating...
Average Loss: 0.410606
Final accuracy: 85.6960%
[INFO]: Epoch 34 of 100
Training ist starting...
Average Loss: 0.125350
Evaluating...
Average Loss: 0.425586
Final accuracy: 85.1560%
[INFO]: Epoch 35 of 100
Training ist starting...
Average Loss: 0.118041
Evaluating...
Average Loss: 0.444294
Final accuracy: 85.4760%
[INFO]: Epoch 36 of 100
Training ist starting...
Average Loss: 0.114334
Evaluating...
Average Loss: 0.443017
Final accuracy: 84.9720%
[INFO]: Epoch 37 of 100
Training ist starting...
Average Loss: 0.103973
Evaluating...
Average Loss: 0.435312
Final accuracy: 85.0280%
[INFO]: Epoch 38 of 100
Training ist starting...
Average Loss: 0.103711
Evaluating...
Average Loss: 0.453689
Final accuracy: 85.4760%
[INFO]: Epoch 39 of 100
Training ist starting...
Average Loss: 0.093727
Evaluating...
Average Loss: 0.470100
Final accuracy: 85.3840%
[INFO]: Epoch 40 of 100
Training ist starting...
Average Loss: 0.092685
Evaluating...
Average Loss: 0.500058
Final accuracy: 84.5480%
[INFO]: Epoch 41 of 100
Training ist starting...
Average Loss: 0.084122
Evaluating...
Average Loss: 0.490171
Final accuracy: 85.0720%
[INFO]: Epoch 42 of 100
Training ist starting...
Average Loss: 0.079752
Evaluating...
Average Loss: 0.507636
Final accuracy: 85.4120%
[INFO]: Epoch 43 of 100
Training ist starting...
Average Loss: 0.072607
Evaluating...
Average Loss: 0.566298
Final accuracy: 84.3640%
[INFO]: Epoch 44 of 100
Training ist starting...
Average Loss: 0.069136
Evaluating...
Average Loss: 0.530423
Final accuracy: 84.5120%
[INFO]: Epoch 45 of 100
Training ist starting...
Average Loss: 0.067138
Evaluating...
Average Loss: 0.538429
Final accuracy: 84.5480%
[INFO]: Epoch 46 of 100
Training ist starting...
Average Loss: 0.060171
Evaluating...
Average Loss: 0.555691
Final accuracy: 84.5480%
[INFO]: Epoch 47 of 100
Training ist starting...
Average Loss: 0.055638
Evaluating...
Average Loss: 0.592749
Final accuracy: 84.6280%
[INFO]: Epoch 48 of 100
Training ist starting...
Average Loss: 0.050464
Evaluating...
Average Loss: 0.599483
Final accuracy: 84.3920%
[INFO]: Epoch 49 of 100
Training ist starting...
Average Loss: 0.051449
Evaluating...
Average Loss: 0.633028
Final accuracy: 84.3880%
[INFO]: Epoch 50 of 100
Training ist starting...
Average Loss: 0.046499
Evaluating...
Average Loss: 0.606237
Final accuracy: 84.5240%
[INFO]: Epoch 51 of 100
Training ist starting...
Average Loss: 0.042994
Evaluating...
Average Loss: 0.653681
Final accuracy: 84.2160%
[INFO]: Epoch 52 of 100
Training ist starting...
Average Loss: 0.041055
Evaluating...
Average Loss: 0.651805
Final accuracy: 84.7680%
[INFO]: Epoch 53 of 100
Training ist starting...
Average Loss: 0.038950
Evaluating...
Average Loss: 0.703097
Final accuracy: 83.9040%
[INFO]: Epoch 54 of 100
Training ist starting...
Average Loss: 0.036206
Evaluating...
Average Loss: 0.678043
Final accuracy: 84.0800%
[INFO]: Epoch 55 of 100
Training ist starting...
Average Loss: 0.036547
Evaluating...
Average Loss: 0.683367
Final accuracy: 84.4040%
[INFO]: Epoch 56 of 100
Training ist starting...
Average Loss: 0.030761
Evaluating...
Average Loss: 0.713800
Final accuracy: 84.1720%
[INFO]: Epoch 57 of 100
Training ist starting...
Average Loss: 0.030594
Evaluating...
Average Loss: 0.711545
Final accuracy: 84.3840%
[INFO]: Epoch 58 of 100
Training ist starting...
Average Loss: 0.032227
Evaluating...
Average Loss: 0.724556
Final accuracy: 84.5440%
[INFO]: Epoch 59 of 100
Training ist starting...
Average Loss: 0.028531
Evaluating...
Average Loss: 0.737367
Final accuracy: 84.2800%
[INFO]: Epoch 60 of 100
Training ist starting...
Average Loss: 0.025582
Evaluating...
Average Loss: 0.749917
Final accuracy: 84.2040%
[INFO]: Epoch 61 of 100
Training ist starting...
Average Loss: 0.027338
Evaluating...
Average Loss: 0.763423
Final accuracy: 84.2080%
[INFO]: Epoch 62 of 100
Training ist starting...
Average Loss: 0.023239
Evaluating...
Average Loss: 0.777951
Final accuracy: 84.3480%
[INFO]: Epoch 63 of 100
Training ist starting...
Average Loss: 0.021617
Evaluating...
Average Loss: 0.792184
Final accuracy: 84.2200%
[INFO]: Epoch 64 of 100
Training ist starting...
Average Loss: 0.020849
Evaluating...
Average Loss: 0.838459
Final accuracy: 84.2000%
[INFO]: Epoch 65 of 100
Training ist starting...
Average Loss: 0.021452
Evaluating...
Average Loss: 0.811581
Final accuracy: 84.2360%
[INFO]: Epoch 66 of 100
Training ist starting...
Average Loss: 0.020773
Evaluating...
Average Loss: 0.816431
Final accuracy: 83.9480%
[INFO]: Epoch 67 of 100
Training ist starting...
Average Loss: 0.021028
Evaluating...
Average Loss: 0.840575
Final accuracy: 83.7320%
[INFO]: Epoch 68 of 100
Training ist starting...
Average Loss: 0.017760
Evaluating...
Average Loss: 0.858958
Final accuracy: 84.1400%
[INFO]: Epoch 69 of 100
Training ist starting...
Average Loss: 0.016165
Evaluating...
Average Loss: 0.861174
Final accuracy: 84.1800%
[INFO]: Epoch 70 of 100
Training ist starting...
Average Loss: 0.017817
Evaluating...
Average Loss: 0.877801
Final accuracy: 83.9440%
[INFO]: Epoch 71 of 100
Training ist starting...
Average Loss: 0.014668
Evaluating...
Average Loss: 0.890617
Final accuracy: 84.0280%
[INFO]: Epoch 72 of 100
Training ist starting...
Average Loss: 0.014476
Evaluating...
Average Loss: 0.904095
Final accuracy: 83.7960%
[INFO]: Epoch 73 of 100
Training ist starting...
Average Loss: 0.013478
Evaluating...
Average Loss: 0.909442
Final accuracy: 83.9400%
[INFO]: Epoch 74 of 100
Training ist starting...
Average Loss: 0.013158
Evaluating...
Average Loss: 0.919251
Final accuracy: 84.0160%
[INFO]: Epoch 75 of 100
Training ist starting...
Average Loss: 0.014046
Evaluating...
Average Loss: 0.916776
Final accuracy: 83.9240%
[INFO]: Epoch 76 of 100
Training ist starting...
Average Loss: 0.014488
Evaluating...
Average Loss: 0.901349
Final accuracy: 84.0760%
[INFO]: Epoch 77 of 100
Training ist starting...
Average Loss: 0.011243
Evaluating...
Average Loss: 0.984252
Final accuracy: 83.7760%
[INFO]: Epoch 78 of 100
Training ist starting...
Average Loss: 0.010653
Evaluating...
Average Loss: 0.964620
Final accuracy: 84.0680%
[INFO]: Epoch 79 of 100
Training ist starting...
Average Loss: 0.012671
Evaluating...
Average Loss: 0.976853
Final accuracy: 83.6800%
[INFO]: Epoch 80 of 100
Training ist starting...
Average Loss: 0.012339
Evaluating...
Average Loss: 0.955030
Final accuracy: 84.0600%
[INFO]: Epoch 81 of 100
Training ist starting...
Average Loss: 0.010245
Evaluating...
Average Loss: 0.996305
Final accuracy: 83.9520%
[INFO]: Epoch 82 of 100
Training ist starting...
Average Loss: 0.010273
Evaluating...
Average Loss: 1.014373
Final accuracy: 83.7640%
[INFO]: Epoch 83 of 100
Training ist starting...
Average Loss: 0.010537
Evaluating...
Average Loss: 1.029586
Final accuracy: 84.0000%
[INFO]: Epoch 84 of 100
Training ist starting...
Average Loss: 0.009604
Evaluating...
Average Loss: 1.039678
Final accuracy: 84.1960%
[INFO]: Epoch 85 of 100
Training ist starting...
Average Loss: 0.011554
Evaluating...
Average Loss: 1.033348
Final accuracy: 83.6440%
[INFO]: Epoch 86 of 100
Training ist starting...
Average Loss: 0.009913
Evaluating...
Average Loss: 1.025658
Final accuracy: 84.1320%
[INFO]: Epoch 87 of 100
Training ist starting...
Average Loss: 0.009683
Evaluating...
Average Loss: 1.045951
Final accuracy: 84.0240%
[INFO]: Epoch 88 of 100
Training ist starting...
Average Loss: 0.010365
Evaluating...
Average Loss: 1.019874
Final accuracy: 83.8320%
[INFO]: Epoch 89 of 100
Training ist starting...
Average Loss: 0.008901
Evaluating...
Average Loss: 1.041410
Final accuracy: 83.7520%
[INFO]: Epoch 90 of 100
Training ist starting...
Average Loss: 0.009086
Evaluating...
Average Loss: 1.080346
Final accuracy: 83.6320%
[INFO]: Epoch 91 of 100
Training ist starting...
Average Loss: 0.007677
Evaluating...
Average Loss: 1.078580
Final accuracy: 83.8160%
[INFO]: Epoch 92 of 100
Training ist starting...
Average Loss: 0.007743
Evaluating...
Average Loss: 1.090401
Final accuracy: 83.6880%
[INFO]: Epoch 93 of 100
Training ist starting...
Average Loss: 0.007800
Evaluating...
Average Loss: 1.077982
Final accuracy: 83.8480%
[INFO]: Epoch 94 of 100
Training ist starting...
Average Loss: 0.005805
Evaluating...
Average Loss: 1.121087
Final accuracy: 83.9920%
[INFO]: Epoch 95 of 100
Training ist starting...
Average Loss: 0.007237
Evaluating...
Average Loss: 1.096096
Final accuracy: 83.6880%
[INFO]: Epoch 96 of 100
Training ist starting...
Average Loss: 0.006814
Evaluating...
Average Loss: 1.079269
Final accuracy: 83.9680%
[INFO]: Epoch 97 of 100
Training ist starting...
Average Loss: 0.006130
Evaluating...
Average Loss: 1.088304
Final accuracy: 83.9680%
[INFO]: Epoch 98 of 100
Training ist starting...
Average Loss: 0.006650
Evaluating...
Average Loss: 1.119643
Final accuracy: 83.7520%
[INFO]: Epoch 99 of 100
Training ist starting...
Average Loss: 0.006792
Evaluating...
Average Loss: 1.125189
Final accuracy: 84.0000%
[INFO]: Epoch 100 of 100
Training ist starting...
Average Loss: 0.006430
Evaluating...
Average Loss: 1.138887
Final accuracy: 83.7240%

'''
# Extrahieren der Daten mit regulären Ausdrücken
train_losses = []
eval_losses = []
accuracies = []

for line in data.split('\n'):
    if line.startswith('Average Loss:'):
        if 'Training' in data.split('\n')[data.split('\n').index(line)-1]:
            train_losses.append(float(line.split(': ')[1]))
        elif 'Evaluating' in data.split('\n')[data.split('\n').index(line)-1]:
            eval_losses.append(float(line.split(': ')[1]))
    elif 'Final accuracy:' in line:
        accuracies.append(float(line.split(': ')[1].strip('%')))

# Erstelle epochs NACH dem Extrahieren der Daten
epochs = range(1, len(train_losses) + 1)

# Plot 1: Training Loss und Validation Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, eval_losses, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Nur Accuracy
plt.figure(figsize=(12, 6))
plt.plot(epochs, accuracies, 'g-', label='Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots wurden als Bilder gespeichert:")
print("1. training_loss.png (mit Training und Validation Loss)")
print("2. accuracy.png")