# Patter Recognition: Nearest Neighbors
import random
import math


def euclidean_distance(horse1, horse2):
    height1, weight1 = horse1
    height2, weight2 = horse2

    return math.sqrt((height1 - height2) ** 2
                     + (weight1 - weight2) ** 2)


def find_nearest_neighbors(test_horse, training_data, k=3):
    distances = []

    for horse in training_data:
        distance = euclidean_distance(test_horse, horse[:2])
        distances.append((distance, horse))

    distances.sort(key=lambda item: item[0])
    nearest_neighbors = [item[1] for item in distances[:k]]

    return nearest_neighbors


def predict(test_horse, training_data, k=3):
    nearest_neighbors = find_nearest_neighbors(test_horse,
                                               training_data, k)

    num_mares = sum(1 for horse in nearest_neighbors
                    if horse[-1] == "mare")

    num_stallions = sum(1 for horse in nearest_neighbors
                        if horse[-1] == "stallion")

    if num_mares > num_stallions:
        return "mare"
    else:
        return "stallion"


random.seed(42)

# Each sample contains:
# [height in cm, weight in kg, class label]

training_data = []

# Generate 100 mares
for _ in range(100):
    height = random.gauss(155, 6)
    weight = random.gauss(450, 45)

    training_data.append([height, weight, "mare"])

# Generate 100 stallions
for _ in range(100):
    height = random.gauss(165, 7)
    weight = random.gauss(520, 50)
    training_data.append([height, weight, "stallion"])

# Shuffle the dataset
random.shuffle(training_data)

print("Total samples:", len(training_data))
print("First five samples:")

for horse in training_data[:5]:
    print(horse)

# New horse: height and weight
# Example: 160, 480
height, weight = map(float, input("Enter height and weight: ").strip().split())
test_horse = [height, weight]

prediction = predict(test_horse, training_data, k=3)
print("Horse features:", test_horse)
print("Predicted class:", prediction)

# Display the nearest neighbors
neighbors = find_nearest_neighbors(test_horse, training_data, k=3)
print("\nNearest neighbors:")

for horse in neighbors:
    print(f"Height: {horse[0]:.1f} cm, "
        f"Weight: {horse[1]:.1f} kg, "
        f"Class: {horse[2]}"
          )
