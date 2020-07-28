# *OpDisc* : A model of discriminatory opinion dynamics

[[_TOC_]]

## Fundamentals
This is an opinion dynamics model simulating the behaviour of several different social groups discriminating against each other.
In the model, the groups are abstract, and their interaction behaviour can be specified by the user; in reality however, these groups will usually
correspond to racial, age, gender, or religious affiliations, which have been shown to be the major determinant of social interaction patterns.
In particular, humans will discriminate *negatively* against members of other groups. This tendency to prefer associating with those who are similar
to oneself is termed *homophily*. Aside from *status homophily* (a propensity to interact with members of similar social groups),
the model also implements *value homophily* (preference for those with similar views) through the usual selective exposure mechanism.

## The *OpDisc* model
The *OpDisc* model is a simple model of opinion dynamics that nonetheless produces a wide range of dynamical behaviour. It consists of *users* (nodes in a network) randomly interacting and updating their *opinions*, which are one-dimensional values in [0, 1]. The underlying network is static, random, and well-mixed –– that is, the probability of any pair of users to interact is constant in time and across the network. The opinion update function depends on the two users' distance in opinion space on the one hand, but also on their group membership. If two users' distance exceeds the *tolerance*, users will not interact, the underlying assumption being that communication is not possible between users whose opinions differ too much. Each user is also assigned a *group*. Members from the same group will always interact constructively; how members from different groups interact depends on the specific *mode* in which the model is run.

### 1––Users
A node in a network. Each user holds an opinion $`\sigma \in [0, 1]`$. Users have a certain *tolerance range* $`\epsilon`$,
and are susceptible to others' opinions. The *susceptibility* $`\mu`$ measures the strength with which users are attraced (or repelled) by a
neighbour's opinion.Finally, each user is assigned a *group* $`n`$, controlled via the `number_of_groups` key.

### 2––Model modes
By specifying the `mode` in the `cfg`, the specific type of discrimination for interactions between members of different groups can be controlled.
The strength of the discrimination is controlled by the *homophily parameter* $`p_{hom}`$ (the `homophily_parameter` key).

- #### `reduced_int_prob`: Reduced interaction probability
  In this mode, the interaction probability for members of different groups is reduced by a factor of $`1-p_{hom}`$.
  If $`p_{hom}=1`$, no interactions between members of different groups are possible.

- #### `reduced_s`: Reduced susceptibility
  Here, the susceptibility for inter-group interactions is reduced by a factor of $`1-p_{hom}`$.
  As before, $`p_{hom}=1`$ entirely inhibits any constructive interaction between members of different groups.

- #### `isolated_1` and `isolated_2`: Isolated discriminators
  In this mode, only certain individuals discriminate against members of other groups; however, those that do discriminate refuse to interact with members of other groups entirely. The proportion of users who discriminate is again controlled via
  the `homophily_parameter`. In the `isolated_1` mode, non-discriminators will still interact with members from other groups even if those members are discriminators. In `isolated_2`, this is no longer possible: here, only members from the same group or non-discriminators can interact.

- #### `conflict_dir` and `conflict_undir`: Directed and undirected conflict
  In these two modes, users can also reject other opinions. In `conflict_undir`, a certain proportion $`p_{d}`$ of the population (the *discriminators*) is universally repelled by opinions from other group members. $`p_{d}`$ can be controlled via the `discriminators` key. Other users continue to interact constructively, though with a reduced susceptibility for members of other groups. This suscepibility reduction is controlled by the `homophily_parameter` key. `Conflict_undir` is the only mode that uses two keys to control the strength and type of discrimination.

  In `conflict_dir`, there is a direction to the discrimination: users with lower group numbers ("younger groups") universally reject opinions from users with higher group numbers ("older groups"). For example, members of group 2 reject opinions from all members of groups 3, 4, 5, ...; older groups interact constructively with younger groups, though with a reduced susceptibility. This suscepibility reduction is as usual controlled by the `homophily_parameter` key.

- #### `ageing`: Ageing
  This mode uses the directed conflict model as its basis, but user groups correspond to an age, which evolves over time. Instead of setting the
  number of groups via `number_of_groups`, set a maximum age via `life_expectancy`. Users are then initialised with a random age between 10 and the life expectancy.
  After each interaction, a user ages by one unit. If a user exceeds the life expectancy, they are reinitialised as a young user of age 10,
  and assigned an opinion from a *parent* (a user between 20 and 40).
  "Older users" are defined as all users whose age is above the user's age plus the `peer_radius`; "younger users" are correspondingly those who are younger than
  the user's age minus the peer radius. The opinion interaction is the same as in `conflict_dir`.

  The time scales of opinion updates and ageing can be chosen by the users, since in general, attitude updating occurs on time scales very different from those of group dynamics (including the ageing process). This can be set using the `time_scale` key.
  
### 3––Update functions
- **Constructive interaction**: as defined [here](https://ts-gitlab.iup.uni-heidelberg.de/utopia/models/-/blob/OpDyn_disregard_model/src/models/OpDisc/utils.hh#L93).
    In each time step, the opinions of two users $`i, j`$ are updated as follows:
   ```math
   \sigma_i (t+1) = \sigma_i(t) + \begin{cases} \mu (\sigma_j(t)-\sigma_i(t)), \ |\sigma_j(t)-\sigma_i(t)| \leq \epsilon, \\ 0, \ \mathrm{else} \end{cases}
   ```
   This happens for *both* users $`i`$ and $`j`$.
- **Rejection interaction**: as defined [here](https://ts-gitlab.iup.uni-heidelberg.de/utopia/models/-/blob/OpDyn_disregard_model/src/models/OpDisc/utils.hh#L83)

   In each time step, the opinions of a user $`i`$ interacting with a user $`j`$ whose opinion he/she rejects is updated as follows (the $`(t)`$ argument is dropped for convenience):

  ```math
  \sigma_i(t+1) = \sigma_i + \begin{cases}   -\mu \sigma_i \dfrac{\sigma_j-\sigma_i}{1-\sigma_i}, \ \mathrm{if} \ \sigma_i < \sigma_j \ \mathrm{or} \ \sigma_i=\sigma_j=0, \\  \mu \left((1-\sigma_i) \dfrac{\sigma_i-\sigma_j}{\sigma_i} \right), \ else \end{cases}
  ```

### 4––Ageing
In each time step, a pair of users update their opinions; their ages are subsequently increased by 1. The interaction will depend on the users age differences,
as defined [here](https://ts-gitlab.iup.uni-heidelberg.de/utopia/models/-/blob/OpDyn_disregard_model/src/models/OpDisc/aging.hh#L27).
If a user's age is higher than the `life_expectancy`, they are *reinitialised* as a child node with age 10, see [here](https://ts-gitlab.iup.uni-heidelberg.de/utopia/models/-/blob/OpDyn_disregard_model/src/models/OpDisc/aging.hh#L9).

### 5––Extremism
If the `extremism` key is set to `true`, users' tolerance will decrease as their opinion moves towards the boundaries of the opinion spectrum. At the boundaries,
the user tolerance will be exactly half the `tolerance` value set in the `model_cfg`.
## Running the model

**Parameters:**

- `nw/num_vertices`:  Sets the number of users.
- `mode`: Defines the discrimination mode. Options are `reduced_int_prob`, `reduced_s`, `isolated_1`, `isolated_2`, `conflict_dir`, `conflict_undir`, `ageing`.
- `number_of_groups`: Sets the number of groups (except for `mode: ageing`).
- `homophily_parameter`: Sets the homophily parameter
- `discriminators`: Sets the proportion of discriminating agents (only in `mode: conflict_undir`).
- `tolerance`: The global user tolerance.
- `susceptibility`: The global user susceptibility.
- `ageing/life_expectancy`: The life expectancy of users in `mode: ageing`.
- `ageing/peer_radius`: The peer radius of users in `mode: ageing`.
- `ageing/time_scale`: The ratio of opinion update to ageing time scale.

## Plots
**Universe Plots:**
- `densities`: Plots the density of opinion clusters over time.
- `group_avgs`: Plots the average opinion of each group over time. See also `group_avgs_anim`.
- `opinion_anim`: Plots an animation of the opinion distribution.
- `opinion_groups`: Plots an animated stacked bar plot of the opinion distribution of each group.

**Multiverse Plots:**
- `bifurcation`: Plots a bifurcation diagramme of the extrema (ie. first derivative=0) of the average opinion over a selected sweep parameter.
- `group_avgs_anim`: Plots an animated plot of the average opinion by group over a selected sweep parameter.

![op_dist](https://ts-gitlab.iup.uni-heidelberg.de/uploads/-/system/user/118/a100df4e2e8d6cfdef2fbaf265cc600f/opinion_distributions.jpeg)
**Fig. 1** `densities` plot (left) and `opinion_anim` plot (right).

![anim](https://ts-gitlab.iup.uni-heidelberg.de/uploads/-/system/user/118/6a8a3ce9a88575556446a994589b17bb/animated_plots.jpeg)
**Fig. 2** `opinion_groups` plot (left) and `group_avgs` plot (right).

![sweep_plt](https://ts-gitlab.iup.uni-heidelberg.de/uploads/-/system/user/118/198cc8437601de0d3b9f8c38f14d5952/sweep_plots.jpeg)
**Fig. 3** `bifurcation` plot (left) and `group_avgs_anim` plot (right).
