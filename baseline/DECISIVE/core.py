"""DECISIVE §§5.2–5.5: numerical inference, acquisition, and recommendation.

No model, simulator, labels, or private persona data enter this module.
"""
from itertools import combinations

import numpy as np

RANKING_PROTOCOL = "decisive_posterior_expected_utility_v1"
QUESTION_PROTOCOL = "decisive_factor_tradeoff_zero_based_v1"


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    positive = p[p > 0]
    return float(-np.sum(positive * np.log2(positive)))


class DecisionModel:
    def __init__(self, matrix, config, seed, *, particles=None):
        self.scores = np.asarray(matrix, dtype=float)
        if (self.scores.ndim != 2 or min(self.scores.shape) < 1
                or not np.isfinite(self.scores).all()
                or np.any((self.scores < 0) | (self.scores > 1))):
            raise ValueError("Option scores must be a nonempty finite matrix in [0,1]")
        self.kappa = config["kappa"]
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be positive and finite")
        self.particles = (np.random.default_rng(seed).dirichlet(
            np.full(self.scores.shape[1], config["dirichlet_alpha"]),
            size=config["particles"]) if particles is None else np.asarray(particles, dtype=float))
        if (self.particles.ndim != 2 or len(self.particles) == 0
                or self.particles.shape[1] != self.scores.shape[1]
                or not np.isfinite(self.particles).all() or np.any(self.particles < 0)
                or not np.allclose(self.particles.sum(axis=1), 1)):
            raise ValueError("Particles must be finite simplex vectors matching the factors")
        self.log_weights = np.full(len(self.particles), -np.log(len(self.particles)))
        self.utilities = self.particles @ self.scores.T
        # The paper leaves argmax ties unspecified. Split tied optimum mass to
        # avoid false confidence from identical options in sparse movie metadata.
        winners = np.isclose(self.utilities, self.utilities.max(axis=1, keepdims=True),
                             rtol=0, atol=1e-12)
        self.winner_mass = winners / winners.sum(axis=1, keepdims=True)

    @property
    def weights(self):
        return np.exp(self.log_weights)

    def decision_distribution(self, weights=None):
        return (self.weights if weights is None else weights) @ self.winner_mass

    def log_likelihood(self, a, b, answer):
        if (type(answer) is not int or answer not in (0, 1) or a == b
                or not 0 <= a < self.scores.shape[1] or not 0 <= b < self.scores.shape[1]):
            raise ValueError("Expected a distinct factor pair and a binary answer")
        delta = self.kappa * (self.particles[:, a] - self.particles[:, b])
        return -np.logaddexp(0, -delta if answer == 0 else delta)

    def posterior(self, a, b, answer):
        log_joint = self.log_weights + self.log_likelihood(a, b, answer)
        log_probability = np.logaddexp.reduce(log_joint)
        return log_joint - log_probability, float(np.exp(log_probability))

    def update(self, a, b, answer):
        self.log_weights, _ = self.posterior(a, b, answer)

    def question_scores(self, excluded=()):
        excluded = {tuple(sorted(pair)) for pair in excluded}
        prior_entropy = entropy(self.decision_distribution())
        result = []
        for a, b in combinations(range(self.scores.shape[1]), 2):
            if (a, b) in excluded:
                continue
            branches = [self.posterior(a, b, answer) for answer in (0, 1)]
            expected_entropy = sum(probability * entropy(
                self.decision_distribution(np.exp(log_weights)))
                for log_weights, probability in branches)
            result.append({"factor_indices": [a, b],
                           "answer_probabilities": [branch[1] for branch in branches],
                           "eig_bits": max(0., prior_entropy - expected_entropy)})
        return result

    def state(self):
        chi = self.decision_distribution()
        return {"particle_weights": self.weights.tolist(),
                "posterior_mean_weights": (self.weights @ self.particles).tolist(),
                "decision_probabilities": chi.tolist(),
                "decision_entropy_bits": entropy(chi),
                "confidence": float(chi.max()),
                "expected_utilities": (self.weights @ self.utilities).tolist()}

    def ranking(self, item_ids):
        if len(item_ids) != self.scores.shape[0] or len(set(item_ids)) != len(item_ids):
            raise ValueError("Item IDs must uniquely match matrix rows")
        utility = self.weights @ self.utilities
        return [item_ids[i] for i in np.argsort(-utility, kind="stable")]
