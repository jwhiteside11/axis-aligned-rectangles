import random
import math

# PAC learner for axis-aligned rectangles
class RectangleLearner:
    def __init__(self, n, m):
        self.rectangle = (0, 1, 0, 1)
        self.data = self.generate_testing_data(self.rectangle, n)
        self.generate(n, m)

    def generate(self, n, m):
        self.training_data = self.generate_training_data(self.rectangle, m)
        self.learned_rectangle = self.learn_axis_aligned_rectangle(self.training_data)
        self.testing_data = self.generate_testing_data(self.rectangle, n)
        self.accuracy = self.get_accuracy(self.rectangle, self.learned_rectangle, self.testing_data)

    def add_point(self):
        r = self.learned_rectangle
        x, y = random.random(), random.random()
        self.training_data.append((x, y))
        self.learned_rectangle = (min(x, r[0]), max(x, r[1]), min(y, r[2]), max(y, r[3]))
        self.accuracy = self.get_accuracy(self.rectangle, self.learned_rectangle, self.testing_data)

    # generates a test dataset where each point has probability of ~0.5 of being in the rectangle
    def generate_testing_data(self, rectangle, n):
        xmin, xmax, ymin, ymax = rectangle
        x_part, y_part = (xmax - xmin) / 5, (ymax - ymin) / 5
        return [(random.uniform(xmin - x_part, xmax + x_part), random.uniform(ymin - y_part, ymax + y_part)) for _ in range(n)]
    
    # generates a traning dataset of positive examples (points in known rectangle)
    def generate_training_data(self, rectangle, m):
        xmin, xmax, ymin, ymax = rectangle
        return [(random.uniform(xmin, xmax), random.uniform(ymin, ymax)) for _ in range(m)]

    # Generate a hypothesis rectangle from randomly assembled training dataset
    def learn_axis_aligned_rectangle(self, training_data):
        xp = [x for x, _ in training_data]
        yp = [y for _, y in training_data]
        # learned rectangle is tightest fit around min/max points found in positive examples
        return (min(xp), max(xp), min(yp), max(yp))

    # Classifies a point as inside (1) or outside (0) the rectangle.
    def classify(self, point, rectangle):
        x, y = point
        xmin, xmax, ymin, ymax = rectangle
        return int(xmin <= x <= xmax and ymin <= y <= ymax)
    
    # Find accuracy of hypothesis by comparing hypothesis positive count with concept
    def get_accuracy(self, rectangle, learned_rect, test_points):
        sum_hx = sum([self.classify(p, learned_rect) for p in test_points])
        sum_cx = sum([self.classify(p, rectangle) for p in test_points])
        return sum_hx / sum_cx


# AdaBoost/bootstrap aggregating learner for axis-aligned rectangles
class AdaboostRectangleLearner:
    # --- AdaBoost :) ---
    def adaboost(self, data, num_rounds=1000, d = 0.005):
        n = len(data)
        weights = [1/n] * n
        classifiers = []

        for t in range(1, num_rounds + 1):
            # Form hypothesis rectangle as tightest fit around 5 randomly sampled positive data points
            rectangle = self.get_random_hypothesis(d)
            # Find weighted error and best polarity - check flipped error for improved accuracy in high variance data
            error, polarity = self.compute_weighted_error(rectangle, data, weights)
            
            # Safety catches
            if error >= 0.5:
                continue  # Skip useless weak learner
            if error == 0:
                error = 1e-10

            # Compute alpha
            alpha = 0.5 * math.log((1 - error) / error)

            # Update weights
            new_weights = []
            for (x, y, label), w in zip(data, weights):
                pred = self.predict_rectangle(x, y, rectangle, polarity)
                new_weight = w * math.exp(alpha if pred != label else -alpha)
                new_weights.append(new_weight)

            # Normalize
            nw_total = sum(new_weights)
            weights = [w / nw_total for w in new_weights]

            # print(f"Round {t}: error={error:.3f}, alpha={alpha:.3f}")
            classifiers.append((rectangle, polarity, alpha))

        # Post-processing: greedily select ensemble with lowest error
        return self.select_best_ensemble_by_alpha(data, classifiers)

    # --- AdaBoost with bootstrap aggregating ---
    def adaboost_with_bagging(self, data, num_bags=1, num_rounds=1, d = 0.25):
        voters = []
        n = len(data)
        bag_size = min(n, int(max(n * 0.1, min(n, 4 * n / num_bags)))) # subsample the data - use minimum 10% of training set

        for _ in range(num_bags):
            # bootstrap dataset
            data_bag = random.choices(data, k=bag_size)
            # perform adaboost to get a boosted ensemble
            ensemble = self.adaboost(data_bag, num_rounds=num_rounds, d=d)
            # add ensemble to voters
            voters.append(ensemble)

        return voters
    
    # --- Find best ensemble greedily by assessing error ---
    def select_best_ensemble_by_alpha(self, data, classifiers, max_ensemble_size=1000):
        # Sort classifiers by their alpha in descending order
        classifiers_sorted = sorted(classifiers, key=lambda x: x[2], reverse=True)

        # Start with empty ensemble
        current_error = 1.0
        ensemble, best_ensemble = [], []
        prev_weights, best_weights = [0.0] * len(data), [0.0] * len(data)

        # Add classifiers greedily by alpha
        for classifier in classifiers_sorted:
            # Stop at the maximum acceptable size of the ensemble
            if len(best_ensemble) >= max_ensemble_size:
                break
            # Compute new ensemble error if this classifier is added
            ensemble.append(classifier)
            new_error = self.compute_ensemble_error(data, ensemble, prev_weights)

            # Only add classifiers that improve the ensemble (reduce the error)
            if new_error < current_error:
                current_error = new_error
                best_ensemble = [*ensemble]
            else:
                # Try just adding this classifier to the existing best
                extra_ensemble, extra_weights = [*best_ensemble, classifier], [*best_weights]
                extra_error = self.compute_ensemble_error(data, extra_ensemble, extra_weights)

                if extra_error < current_error:
                    current_error = extra_error
                    best_ensemble = extra_ensemble
                    best_weights = extra_weights

        # print(f"Ensemble Error: {current_error:.3f}")
        return best_ensemble

    # Generate n data points with an optional noise rate
    def generate_data(self, n=10000, noise_rate=0):
        data = []
        # Create concept rect for the training data to be generated
        concept_rectangle = self.generate_concept_rectangle()
        noise_idxs = set(random.sample([i for i in range(n)], k=int(n*noise_rate)))
        for i in range(n):
            # Generate random points between 0 and 1
            x, y = random.uniform(0, 1), random.uniform(0, 1)
            # Label data based on the concept's position
            true_label = self.classify(concept_rectangle, x, y)
            # Flip the label to incorporate noise (false positive or false negative)
            data.append((x, y, true_label if i not in noise_idxs else 1 - true_label))
        
        return data, concept_rectangle  # Return the training data and the concept rectangle
    
    # --- Rectangle (4-tuple) helpers ---

    def generate_concept_rectangle(self):
        return self.get_random_rectangle_ab(0.3, 0.6)

    # Generate a hypothesis rectangle randomly ith E[A] = 0.5 - d
    def get_random_hypothesis(self, d = 0.25):
        # E[A] = 0.5 - d = E[w] * E[h]
        # E[w] = E[h] = √(0.5 - d)
        d = max(0, min(d, 0.5)) # 0 < d < 0.5
        if d > 0.25:
            x = 2 * ((0.5 - d) ** 0.5)
            return self.get_random_rectangle_ab(0.0, x)
        else:
            x = 2 * ((0.5 - d) ** 0.5) - 1
            return self.get_random_rectangle_ab(x, 1.0)
    
    # Generate a random rectangle w/ side lengths distributed over (a, b)
    def get_random_rectangle_ab(self, a, b):
        w = random.uniform(a, b)
        h = random.uniform(a, b)
        x1, y1 = random.uniform(0.0, 1.0 - w), random.uniform(0.0, 1.0 - h)
        # Rectangle of form (x1, x2, y1, y2)
        return (x1, x1 + w, y1, y1 + h)
    
    # Find the hypothesis reached by the boosted & bagged strong learner (purple)
    def get_consensus_rectangle(self, voters):
        n = len(voters)
        resolution = 200
        positive_points = [] * n

        for i in range(resolution):
            for j in range(resolution):
                x = i / resolution
                y = j / resolution
                if self.boosted_bagged_predict(x, y, voters) == 1:
                    positive_points.append((x, y))

        if not positive_points:
            return None
            
        xs, ys = zip(*positive_points)
        return (min(xs), max(xs), min(ys), max(ys))

    # --- Error helpers ---

    def compute_weighted_error(self, rectangle, data, weights):
        error, flipped_error = 0.0, 0.0
        for (x, y, label), w in zip(data, weights):
            pred = self.predict_rectangle(x, y, rectangle, 1)
            if pred != label:
                error += 1 * w if label == 1 else 1.05 * w # penalize false negatives extra
            # check flipped 
            pred = self.predict_rectangle(x, y, rectangle, -1)
            if pred != label:
                flipped_error += 1 * w if label == 1 else 1.05 * w # penalize false negatives extra

        if flipped_error < error:
            return min(1.0, flipped_error), -1
        else:
            return min(1.0, error), 1

    def compute_ensemble_error(self, data, ensemble, previous_weights):
        total_error = 0.0
        n = len(data)
        
        rectangle, polarity, alpha = ensemble[-1]

        for i in range(n):
            (x, y, label) = data[i]
            pred = self.predict_rectangle(x, y, rectangle, polarity)
            previous_weights[i] += alpha * (2 * pred - 1)  # pred = 0 or 1, so 2 * pred - 1 gives -1 or 1

            # If the weighted sum is positive, classify as 1, else as 0
            predicted_label = 1 if previous_weights[i] > 0 else 0
            if predicted_label != label:
                total_error += 1

        return total_error / n
    
    def test_error(self, data, hypothesis):
        if not hypothesis:
            return 1.0
        
        n = len(data)
        total_error = 0
        for i in range(n):
            (x, y, label) = data[i]
            pred = self.classify(hypothesis, x, y)
            if pred != label:
                total_error += 1

        return total_error / n
    
    def true_error(self, concept, hypothesis):
        if not hypothesis:
            return 1.0, 0.0
        
        x1_t, x2_t, y1_t, y2_t = concept
        x1_h, x2_h, y1_h, y2_h = hypothesis

        A_true = abs((x2_t - x1_t) * (y2_t - y1_t))
        A_hyp  = abs((x2_h - x1_h) * (y2_h - y1_h))

        x1_int = max(x1_t, x1_h)
        x2_int = min(x2_t, x2_h)
        y1_int = max(y1_t, y1_h)
        y2_int = min(y2_t, y2_h)

        if x1_int < x2_int and y1_int < y2_int:
            A_int = (x2_int - x1_int) * (y2_int - y1_int)
        else:
            A_int = 0.0

        return A_true + A_hyp - 2 * A_int, A_true

    
    # --- Prediction helpers ---

    def classify(self, rect, x, y):
        xmin, xmax, ymin, ymax = rect
        return 1 if xmin <= x <= xmax and ymin <= y <= ymax else 0
    
    def predict_rectangle(self, x, y, rect, polarity):
        pred = self.classify(rect, x, y)
        return 1 - pred if polarity == -1 else pred
    
    def boosted_predict(self, x, y, classifiers):
        total = 0.0
        for (rectangle, polarity, alpha) in classifiers:
            total += alpha if self.predict_rectangle(x, y, rectangle, polarity) == 1 else -alpha  # Boosted vote: +alpha if 1, -alpha if 0
        return 1 if total > 0 else 0
    
    def boosted_bagged_predict(self, x, y, voters):
        total = 0
        for ensemble in voters:
            total += self.boosted_predict(x, y, ensemble)

        n = len(voters)
        return 1 if total > n / 2 else 0
    

if __name__ == "__main__":
    learner = AdaboostRectangleLearner()

    err = 0.0
    ov = 0.0
    n = 1000000
    for i in range(n):
        concept_rectangle = learner.generate_concept_rectangle()
        random_hypothesis = learner.get_random_hypothesis()
        err_, ov_ = learner.true_error(concept_rectangle, random_hypothesis)
        err += err_
        ov += ov_

    print(f"Average error: {err / n:.6f} {ov / n:.6f}")