# Example images

Downloaded from Wikimedia Commons by `scripts/fetch_examples.py`; author and
licence are read from the Commons API. All images were **resized** so that
the longest side is 600 px and re-encoded as JPEG; no other change.

| File | Use | Source | Author | Licence |
|---|---|---|---|---|
| `forbidden_city_smog.jpg` (600×546) | speed benchmark (a hard, dense-haze case, not shown as an "it works" example) | [Beijing Forbidden City Smog.jpg](https://commons.wikimedia.org/wiki/File:Beijing_Forbidden_City_Smog.jpg) | Brian Jeffery Beggerly | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) |
| `snow_fog.jpg` (600×398) | failure case: bright surfaces (snow) | [Winter fog is coming (31246745274).jpg](https://commons.wikimedia.org/wiki/File:Winter_fog_is_coming_(31246745274).jpg) | marsupium photography | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0) |
| `sunset_sky.jpg` (600×371) | failure case: coloured, non-uniform illumination | [Crepuscular rays color.jpg](https://commons.wikimedia.org/wiki/File:Crepuscular_rays_color.jpg) | PiccoloNamek | [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/) |
| `temple.jpg` (600×450) | qualitative example: the paper's own featured teaser image | [Tiananmen (input haze image, He, Sun & Tang, CVPR 2009 project page)](https://people.csail.mit.edu/kaiming/cvpr09/tiananmen/tiananmen1.png) | unknown | The primary demo image on the paper's own project homepage, shown there as 'input haze image' alongside the authors' own result. Hosted by Kaiming He, one of the paper's authors. |
| `morning_mist_forest.jpg` (600×400) | qualitative example: forest with light mist and clear depth layers | [Gum trees in the early Morning mist at Sheepyard flat (6127021798).jpg](https://commons.wikimedia.org/wiki/File:Gum_trees_in_the_early_Morning_mist_at_Sheepyard_flat_(6127021798).jpg) | Takver from Australia | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0) |
| `toys.jpg` (500×360) | pipeline figure: the paper's own demo image | [Toys](https://people.csail.mit.edu/kaiming/cvpr09/toys/toys.jpg) | unknown | Hosted on the CVPR 2009 project page of Kaiming He, one of the paper's authors, and used there as a benchmark input image. No explicit licence is stated on the page; used here for research/educational reproduction of the paper it comes from. |
| `ville.jpg` (332×500) | qualitative example: dense repeated signage, many small colour patches | [unknown (street with shop signage, likely Kathmandu, Nepal)]() | unknown | ORIGIN AND LICENCE NOT VERIFIED. Not found on He et al.'s or Fattal's project pages (the two sources checked). May be subject to copyright. Kept at the repository author's own discretion pending a reverse-image search. |

None of these images has a haze-free reference: they are used for
qualitative illustration only. Quantitative results come from
`scripts/synthetic_experiment.py` (known ground truth) and
`scripts/benchmark_dataset.py` (paired datasets).
