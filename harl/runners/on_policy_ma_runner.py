"""MAPPO updates on a common, frozen cascade rollout."""
from harl.common.advantages import advantage_mode, all_row_advantages
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyMARunner(OnPolicyBaseRunner):
    """Update shared or independent MAPPO policies."""

    def train(self):
        """Use own mixed-return advantages and a fixed-scale critic update."""
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
        if advantages.shape[-1] != self.num_agents:
            raise ValueError('Expected one mixed return per actor')
        if advantage_mode(self.algo_args['algo']) == 'all_rows':
            advantages = all_row_advantages(advantages)

        own_advantages = [advantages[..., i:i+1].copy() for i in range(self.num_agents)]
        options = {'diagnostics': True} if getattr(self, 'diagnostics_enabled', False) else {}

        actor_train_infos = []
        if self.algo_args['algo'].get('share_param', False):
            core = getattr(self.actor[0], 'core', None)
            if core is None or any(getattr(actor, 'core', None) is not core for actor in self.actor):
                raise ValueError('Shared MAPPO requires agent views of one common policy core')
            # All five own-return channels enter ONE joint dataset and ONE
            # optimizer lifecycle. Do not call each agent view's train method.
            shared_info = core.share_param_train(self.actor_buffer, own_advantages,
                                                self.num_agents, 'EP', **options)
            # Existing logs expect five actor rows. These are repeated shared
            # update statistics, not five separate optimizer-step counts.
            actor_train_infos = [dict(shared_info, shared_actor=1.,
                shared_actor_group_size=float(self.num_agents), shared_actor_info_repeated=1.)
                for _ in range(self.num_agents)]
        else:
            # Retain the independent low-level MAPPO variant for existing
            # callers; every optimizer still uses frozen rollout probabilities.
            for agent_id in range(self.num_agents):
                info = self.actor[agent_id].train(self.actor_buffer[agent_id],
                                                own_advantages[agent_id], 'EP', **options)
                actor_train_infos.append(info)

        # Own advantages and actor updates use the rollout's original moments.
        # Update each value channel once from all rollout returns, then
        # retain that scale for every critic epoch and minibatch.
        critic_train_info = self.critic.train(
            self.critic_buffer, self.value_normalizer, update_normalizer_once=True
        )
        return actor_train_infos, critic_train_info
