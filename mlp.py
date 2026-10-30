import numpy as np
import pandas as pd


# change these if you want to experiment
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.1
SEED = 42


def load_data(path):
    # first column is the answer, the rest are pixel values
    data = pd.read_csv(path).to_numpy()
    labels = data[:, 0].astype(int)
    images = data[:, 1:].astype(np.float32) / 255.0
    return images, labels


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    # shift the values so the exponentials stay manageable
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class TinyDigitBrain:
    def __init__(self, seed=SEED):
        rng = np.random.default_rng(seed)

        # three layers: pixels -> hidden layer -> hidden layer -> digit guess
        self.w1 = rng.normal(0, np.sqrt(2 / 784), (784, 128))
        self.b1 = np.zeros((1, 128))

        self.w2 = rng.normal(0, np.sqrt(2 / 128), (128, 64))
        self.b2 = np.zeros((1, 64))

        self.w3 = rng.normal(0, np.sqrt(1 / 64), (64, 10))
        self.b3 = np.zeros((1, 10))

    def predict_proba(self, images):
        hidden1 = relu(images @ self.w1 + self.b1)
        hidden2 = relu(hidden1 @ self.w2 + self.b2)
        return softmax(hidden2 @ self.w3 + self.b3)

    def predict(self, images):
        return np.argmax(self.predict_proba(images), axis=1)

    def train_batch(self, images, labels):
        batch_size = len(images)
        answers = np.eye(10)[labels]

        # forward pass
        z1 = images @ self.w1 + self.b1
        a1 = relu(z1)

        z2 = a1 @ self.w2 + self.b2
        a2 = relu(z2)

        probs = softmax(a2 @ self.w3 + self.b3)

        # cross-entropy loss
        loss = -np.mean(np.sum(answers * np.log(probs + 1e-9), axis=1))

        # backprop: figure out how much each weight contributed to the error
        dz3 = (probs - answers) / batch_size
        dw3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        dz2 = (dz3 @ self.w3.T) * (z2 > 0)
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ self.w2.T) * (z1 > 0)
        dw1 = images.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # take a small step in the direction that reduces the error
        self.w3 -= LEARNING_RATE * dw3
        self.b3 -= LEARNING_RATE * db3
        self.w2 -= LEARNING_RATE * dw2
        self.b2 -= LEARNING_RATE * db2
        self.w1 -= LEARNING_RATE * dw1
        self.b1 -= LEARNING_RATE * db1

        return loss

    def save(self, path="tiny_digit_brain.npz"):
        # save the learned weights so you can use them later
        np.savez(
            path,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3,
        )


def main():
    print("loading handwritten digits...")

    x_train, y_train = load_data("mnist_train.csv")
    x_test, y_test = load_data("mnist_test.csv")

    brain = TinyDigitBrain()
    rng = np.random.default_rng(SEED)

    print(f"training on {len(x_train):,} images")
    print("-" * 55)

    for epoch in range(EPOCHS):
        # shuffle the images so each epoch gets a different order
        order = rng.permutation(len(x_train))
        losses = []

        for start in range(0, len(x_train), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = brain.train_batch(x_train[batch], y_train[batch])
            losses.append(loss)

        guesses = brain.predict(x_test)
        accuracy = np.mean(guesses == y_test)

        # little progress bar, because plain numbers are kind of boring
        filled = int(accuracy * 20)
        bar = "█" * filled + "░" * (20 - filled)

        print(
            f"epoch {epoch + 1:02}/{EPOCHS}  "
            f"loss {np.mean(losses):.4f}  "
            f"test [{bar}] {accuracy:.1%}"
        )

    brain.save()
    print("-" * 55)
    print("done! saved the model as tiny_digit_brain.npz")


if __name__ == "__main__":
    main()
