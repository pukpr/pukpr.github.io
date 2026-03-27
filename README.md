# pukpr.github.io

**Live site:** https://pukpr.github.io

## About

This repository is the source for a static website that is **automatically built and published by [GitHub Pages](https://pages.github.com/)**.  Every time a commit is pushed to the default branch, GitHub's built-in Jekyll process regenerates the site from the files in this repository and deploys it to the URL above — no manual build or deployment step is required.

The site is rendered using the [`jekyll-theme-leap-day`](https://github.com/pages-themes/leap-day) theme and is configured in [`_config.yml`](./_config.yml).

## Contents

The website is Paul Pukite's collection of climate and geosciences resources, including:

- **Climate index and sea-level modelling** — interactive results for PSMSL tide-gauge stations and major climate indices, driven by the [GEM LTE](https://github.com/pukpr/GEM-LTE) Laplace Tidal Equation model.
- **Worked examples** — introductory notebooks and demos (Warnemünde, PySINDy latent-layer oscillation, multiple linear regression).
- **Research paper** — *The modelled climatic response to the 18.6-year lunar nodal cycle and its role in decadal temperature trends* (included as a PDF).
- **External links** — the [GeoEnergyMath blog](https://GeoEnergyMath.com), the [Azimuth Project](https://azimuth-project.github.io) climate-modelling wiki, the [Mathematical Geoenergy](https://agupubs.onlinelibrary.wiley.com/doi/book/10.1002/9781119434351) AGU book, and the [Peak Oil Barrel](https://PeakOilBarrel.com) depletion-modelling blog.

## Repository layout

| Path | Description |
|------|-------------|
| `index.html` | Home page (Jekyll front matter + HTML) |
| `_config.yml` | Jekyll site configuration (theme, title, description) |
| `_data/` | Data files consumed by Jekyll templates |
| `results/` | Generated HTML result pages |
| `examples/` | Example notebooks and demos |
| `*.pdf` | Archived research paper |

## Further reading

- [GeoEnergyMath wiki](https://github.com/pukpr/GeoEnergyMath/wiki) — extended documentation and background theory
- [GEM LTE source repository](https://github.com/pukpr/GEM-LTE) — the underlying tidal model code

