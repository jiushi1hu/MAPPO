import unittest

import torch

from harl.models.base.distributions import OrderedGaussian


class OrderedActorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = OrderedGaussian(4, 100, "orthogonal_", .01)

    def test_ordered_probabilities_and_exploration(self):
        x = torch.zeros(1, 4)
        for bias in (-10., -1., 0., 1., 10.):
            with torch.no_grad():
                self.head.parameters_head.bias[0] = bias
            dist = self.head(x)
            mode = int(dist.mode())
            p = dist.probs[0]
            self.assertTrue((p[1:mode+1] >= p[:mode]).all())
            self.assertTrue((p[mode+1:] <= p[mode:-1]).all())
            self.assertAlmostEqual(float(p.sum()), 1., places=6)
        with torch.no_grad():
            self.head.parameters_head.bias.zero_()
        dist = self.head(x.expand(20000, -1))
        sampled = dist.sample()
        self.assertEqual(sampled.shape, (20000, 1))
        bins = sampled.flatten()
        self.assertGreater(bins.unique().numel(), 95)
        self.assertAlmostEqual(float(bins.float().mean()), 49.5, delta=.6)
        expected = dist.probs.gather(1, sampled).log()
        torch.testing.assert_close(dist.log_probs(sampled), expected)
        torch.testing.assert_close(dist.entropy(), -(dist.probs * dist.logits).sum(-1))
        self.assertGreater(float(dist.entropy()[0]), 4.4)

    def test_masks_and_gradients(self):
        x = torch.ones(3, 4)
        mask = torch.ones(3, 100)
        mask[:, :10] = 0
        dist = self.head(x, mask)
        self.assertTrue((dist.probs[:, :10] == 0).all())
        loss = -(dist.log_probs(torch.tensor([[10], [27], [48]])).mean() + .001*dist.entropy().mean())
        loss.backward()
        grads = self.head.parameters_head.weight.grad
        self.assertTrue(torch.isfinite(grads).all())
        self.assertTrue((grads.norm(dim=1) > 0).all())
        with self.assertRaises(ValueError):
            self.head(x, torch.zeros_like(mask))


if __name__ == '__main__':
    unittest.main()
