# DiffusEdit

> The code and documentation for this project can be found
> [here](https://github.com/gamemaker1/diffusedit).

Reference articles go out of date as events develop, and updating them by hand
is slow. Regenerating the article with a language model automates the update,
but it rewrites text that was already correct and leaves no record of what
changed. DiffusEdit instead edits the article in place. Small trained layers,
reading the hidden states of a frozen masked diffusion language model (LLaDA),
decide which sentences to keep, edit, or delete, and which tokens inside them
are stale. The frozen model rewrites only the stale positions, and the system
repeats the process on sentences that the rewrites themselves made wrong.
Training labels come from pairs of Wikipedia snapshots (FRUIT-Wiki), and
evaluation measures update quality, faithfulness, minimality, and coverage of
the cascading edits.
