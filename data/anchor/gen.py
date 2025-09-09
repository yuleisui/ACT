with open("mnist_csv.csv", 'w') as f:
    for i in range(784):
        f.write(f"{i},")