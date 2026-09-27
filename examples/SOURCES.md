# Example images

Downloaded from Wikimedia Commons by `scripts/fetch_examples.py`; author and
licence are read from the Commons API. All images were **resized** so that
the longest side is 600 px and re-encoded as JPEG; no other change.

| File | Use | Source | Author | Licence |
|---|---|---|---|---|
| `forbidden_city_smog.jpg` (600×546) | pipeline figure, timing benchmark | [Beijing Forbidden City Smog.jpg](https://commons.wikimedia.org/wiki/File:Beijing_Forbidden_City_Smog.jpg) | Brian Jeffery Beggerly | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) |
| `snow_fog.jpg` (600×398) | failure case: bright surfaces (snow) | [Winter fog is coming (31246745274).jpg](https://commons.wikimedia.org/wiki/File:Winter_fog_is_coming_(31246745274).jpg) | marsupium photography | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0) |
| `sunset_sky.jpg` (600×371) | failure case: coloured, non-uniform illumination | [Crepuscular rays color.jpg](https://commons.wikimedia.org/wiki/File:Crepuscular_rays_color.jpg) | PiccoloNamek | [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/) |
| `aerial_perspective_hills.jpg` (600×450) | qualitative example: classic atmospheric perspective (haze increasing with distance) | [Aerial perspective 1.JPG](https://commons.wikimedia.org/wiki/File:Aerial_perspective_1.JPG) | Joaquim Alves Gaspar | [CC BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5) |
| `morning_mist_forest.jpg` (600×400) | qualitative example: forest with light mist and clear depth layers | [Gum trees in the early Morning mist at Sheepyard flat (6127021798).jpg](https://commons.wikimedia.org/wiki/File:Gum_trees_in_the_early_Morning_mist_at_Sheepyard_flat_(6127021798).jpg) | Takver from Australia | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0) |

None of these images has a haze-free reference: they are used for
qualitative illustration only. Quantitative results come from
`scripts/synthetic_experiment.py` (known ground truth) and
`scripts/benchmark_dataset.py` (paired datasets).
